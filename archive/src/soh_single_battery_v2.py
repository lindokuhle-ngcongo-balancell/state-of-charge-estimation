
"""
soh_single_battery.py
─────────────────────
Fetch the full history of a single battery from Athena, estimate per-cell SOC
from a CSV/OCV voltage lookup table with resistance compensation, and derive
per-cell SOH from the SOC delta.

This is the SOH version of soc_reported_vs_estimate_resistance_compensation.py:
    • LUT is built from constant-current CSV files plus optional OCV CSV.
    • Actual current is kept continuous and matched to the nearest LUT bin.
    • Voltage is IR-compensated before the lookup for non-OCV points.
    • SOC reference can be the reported/charge_top SOC or the old per-cell q_top SOC.
    • SOC/SOH can be updated on all points or only at OCV/idle points.

Usage
-----
uv run soh_single_battery.py \
    --serial P34-000092 \
    --capacity 277 \
    --num_cells 32 \
    --csv_dir data/csv_lut \
    --ocv_csv data/ocv_data_for_plotting_q.csv \
    --resistance 0.0004 \
    --soc_source reported \
    --soc_update_mode all \
    --current_threshold -100 \
    --aws_profile infra_dev \
    --output_csv soh_all_updates.csv

OCV-only comparison version:
uv run src/soh_single_battery_v2.py --serial BE19090008  --capacity 277 --num_cells 16 --csv_dir data/csv_lut --ocv_csv data/ocv_data_for_plotting_q.csv --soc_source reported  --soc_update_mode all --output_csv soh_ocv_only_updates.csv
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Literal

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 1441
WINDOW_SIZE_SOC = 7
LUT_REFERENCE_CAPACITY = 277.0
NOMINAL_RESISTANCE = 0.0004
NEGATIVE_LIMIT = -5.0
POSITIVE_LIMIT = 5.0

SocSource = Literal["reported", "per_cell"]
SocUpdateMode = Literal["all", "ocv_only"]


# ── Weights ───────────────────────────────────────────────────────────────────
def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))
    return (weights / weights.sum()).tolist()


WEIGHTS_SOC = make_weights(WINDOW_SIZE_SOC, decay=0.7)
WEIGHTS_SOH = make_weights(WINDOW_SIZE, decay=0.7)


# ── Build LUT from CSV files ──────────────────────────────────────────────────
def _parse_csv_filename(fname: str) -> tuple[str, int] | None:
    """Parse Charging/Discharging current from bundled single-cell CSV names."""
    m = re.search(r"(Charging|Discharging)_Current_(\d+)A\.csv$", fname, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).capitalize(), int(m.group(2))


def build_lut_from_csvs(
    csv_dir: str,
    capacity_ah: float,
    ocv_csv: str | None = None,
) -> pl.DataFrame:
    """
    Build a voltage-DOD LUT from constant-current CSVs and optional OCV CSV.

    Output columns:
        current_bin        signed A, binned to nearest lower 5 A
        dod                0-100 %
        cell_voltages_mean mV
    """
    rows: list[pd.DataFrame] = []
    for fname in sorted(os.listdir(csv_dir)):
        parsed = _parse_csv_filename(fname)
        if parsed is None:
            continue

        direction, current_abs = parsed
        sign = 1 if direction == "Charging" else -1
        current_raw = sign * current_abs
        current_scaled = current_raw / capacity_ah * LUT_REFERENCE_CAPACITY
        current_bin = float((current_scaled // 5) * 5)

        if NEGATIVE_LIMIT <= current_bin <= POSITIVE_LIMIT and ocv_csv is not None:
            print(f"  Skipping {fname} (near-0 A bin superseded by OCV file)")
            continue

        df = pd.read_csv(os.path.join(csv_dir, fname))
        df["soc_int"] = df["soc"].round(0).astype(int)
        df["volt_mv"] = (df["voltage"] * 1000).round(0)

        df_g = (
            df.groupby("soc_int")["volt_mv"]
            .mean()
            .round(0)
            .reset_index()
        )
        df_g["dod"] = (100 - df_g["soc_int"]).astype(float)
        df_g["current_bin"] = 0.0 if NEGATIVE_LIMIT <= current_bin <= POSITIVE_LIMIT else current_bin
        df_g["cell_voltages_mean"] = df_g["volt_mv"]
        rows.append(df_g[["current_bin", "dod", "cell_voltages_mean"]])

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined = (
            combined
            .groupby(["current_bin", "dod"], as_index=False)["cell_voltages_mean"]
            .mean()
            .round(0)
        )
    else:
        combined = pd.DataFrame(columns=["current_bin", "dod", "cell_voltages_mean"])

    if ocv_csv is not None:
        df_ocv = pd.read_csv(ocv_csv)
        df_ocv = df_ocv[df_ocv["soh"] == 100.0].copy()
        if df_ocv.empty:
            raise ValueError(f"OCV file '{ocv_csv}' contains no rows with soh=100.")

        df_ocv["dod"] = (df_ocv["q_removed_ah_ff"] / capacity_ah * 100).round(0)
        df_ocv["volt_mv"] = (df_ocv["cell_voltages"] * 1000).round(0)
        df_ocv_g = (
            df_ocv.groupby("dod", as_index=False)["volt_mv"]
            .mean()
            .round(0)
            .rename(columns={"volt_mv": "cell_voltages_mean"})
        )
        df_ocv_g["current_bin"] = 0.0
        combined = pd.concat(
            [combined, df_ocv_g[["current_bin", "dod", "cell_voltages_mean"]]],
            ignore_index=True,
        )
        print(f"  OCV file loaded: {len(df_ocv_g)} rows for 0 A bin (SOH=100)")

    if combined.empty:
        raise FileNotFoundError(
            f"No LUT data found. Check csv_dir='{csv_dir}' and ocv_csv='{ocv_csv}'."
        )

    lut = pl.from_pandas(combined).sort(["current_bin", "dod"])
    bins = sorted(lut["current_bin"].unique().to_list())
    print(f"LUT ready: {len(lut):,} rows across {len(bins)} current bins: {bins}")
    return lut


# ── SOC estimation with IR compensation and nearest-bin matching ──────────────
def vectorised_soc_estimation(
    lut: pl.DataFrame,
    df: pl.DataFrame,
    resistance_ohm: float,
) -> pl.DataFrame:
    """
    Estimate SOC from voltage by nearest-current-bin lookup.

    Required df columns:
        current_actual   continuous signed current, reference-capacity-scaled A
        cell_voltages    mV
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("soc_min"),
            pl.lit(None, dtype=pl.Float64).alias("soc_max"),
            pl.lit(None, dtype=pl.Float64).alias("nearest_current_bin"),
            pl.lit(None, dtype=pl.Float64).alias("voltage_adjusted_mv"),
            pl.lit(False).alias("is_ocv_update"),
        )

    r_mv = resistance_ohm * 1000.0

    lut_bins = np.sort(lut["current_bin"].unique().to_numpy())
    lut_curr = lut["current_bin"].to_numpy()
    lut_dod = lut["dod"].to_numpy()
    lut_volt = lut["cell_voltages_mean"].to_numpy()

    currents_actual = df["current_actual"].to_numpy()
    voltages = df["cell_voltages"].to_numpy()
    n = len(df)

    soc_min_arr = np.full(n, np.nan)
    soc_max_arr = np.full(n, np.nan)

    bin_indices = np.searchsorted(lut_bins, currents_actual)
    bin_indices = np.clip(bin_indices, 0, len(lut_bins) - 1)
    left_idx = np.clip(bin_indices - 1, 0, len(lut_bins) - 1)
    right_idx = bin_indices
    dist_left = np.abs(currents_actual - lut_bins[left_idx])
    dist_right = np.abs(currents_actual - lut_bins[right_idx])
    nearest_bins = np.where(dist_left <= dist_right, lut_bins[left_idx], lut_bins[right_idx])

    current_delta = currents_actual - nearest_bins
    is_ocv_update = (
        (nearest_bins >= NEGATIVE_LIMIT)
        & (nearest_bins <= POSITIVE_LIMIT)
        & (currents_actual >= NEGATIVE_LIMIT)
        & (currents_actual <= POSITIVE_LIMIT)
    )
    v_adjusted = np.where(is_ocv_update, voltages, voltages + r_mv * current_delta)

    for curr_bin in np.unique(nearest_bins):
        lut_mask = lut_curr == curr_bin
        if not lut_mask.any():
            continue

        bin_dod = lut_dod[lut_mask]
        bin_volt = lut_volt[lut_mask]
        valid = bin_volt >= 0
        bin_dod = bin_dod[valid]
        bin_volt = bin_volt[valid]
        if len(bin_dod) == 0:
            continue

        sort_idx = np.argsort(bin_dod)
        bin_dod = bin_dod[sort_idx]
        bin_volt = bin_volt[sort_idx]

        v_max = bin_volt[0]
        v_min = bin_volt[-1]
        data_mask = nearest_bins == curr_bin
        v_query = v_adjusted[data_mask]

        flipped_volt = bin_volt[::-1]
        flipped_dod = bin_dod[::-1]
        idx_right = np.searchsorted(flipped_volt, v_query, side="left").clip(0, len(flipped_volt) - 1)
        idx_left = (idx_right - 1).clip(0, len(flipped_volt) - 1)

        dod_at_left = flipped_dod[idx_left]
        dod_at_right = flipped_dod[idx_right]
        dod_min_v = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
        dod_max_v = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

        above = v_query > v_max
        below = v_query < v_min
        dod_min_v = np.where(above, 0, np.where(below, 100, dod_min_v))
        dod_max_v = np.where(above, 0, np.where(below, 100, dod_max_v))

        soc_min_arr[data_mask] = (100 - dod_max_v).clip(0, 100)
        soc_max_arr[data_mask] = (100 - dod_min_v).clip(0, 100)

    return df.with_columns([
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
        pl.Series("nearest_current_bin", nearest_bins, dtype=pl.Float64),
        pl.Series("voltage_adjusted_mv", v_adjusted, dtype=pl.Float64),
        pl.Series("is_ocv_update", is_ocv_update, dtype=pl.Boolean),
    ])


# ── Athena fetch ──────────────────────────────────────────────────────────────
def build_query(serial: str, num_cells: int, capacity_ah: float, start_time: str) -> str:
    capacity_coulombs = capacity_ah * 3600.0
    return f"""
SELECT DISTINCT
    device_serial,
    sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current,
    charge_in,
    charge_out,
    charge_top,
    cell_data.cell_voltage AS cell_voltages,
    ((({capacity_coulombs} - (charge_top - (charge_in - charge_out))) / {capacity_coulombs}) * 100) AS soc_reported,
    temperature_battery
FROM analytics.batteries_v3
WHERE device_serial = '{serial}'
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '{start_time}'
ORDER BY sequence_number ASC
"""


def fetch_data(
    serial: str,
    num_cells: int,
    capacity_ah: float,
    aws_profile: str,
    start_time: str,
) -> pl.DataFrame:
    boto3.setup_default_session(profile_name=aws_profile)
    sql = build_query(serial, num_cells, capacity_ah, start_time)
    print(f"Fetching data for {serial} from Athena …")
    pd_df = wr.athena.read_sql_query(sql=sql, database="scratch", workgroup="data_engineering")
    return pl.from_pandas(pd_df)


# ── SOC reference preparation ────────────────────────────────────────────────
def compute_per_cell_soc(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """Old q_top based per-cell SOC reference retained for comparison."""
    capacity_coulombs = capacity_ah * 3600.0

    return (
        df_raw
        .sort(["device_serial", "sequence_number"])
        .with_columns(
            pl.lit(list(range(num_cells))).cast(pl.List(pl.Int16)).alias("cell_position"),
            (pl.col("cell_voltages") * 1000).cast(pl.List(pl.Int32)),
            (pl.col("charge_in") - pl.col("charge_out")).alias("delta_charge"),
            ((pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0)
            .over("device_serial", order_by="sequence_number")
            .alias("current_average_raw"),
        )
        .with_columns(
            pl.when(pl.col("current_average_raw") > 1).then(1)
            .when(pl.col("current_average_raw") < -1).then(0)
            .otherwise(2)
            .cast(pl.UInt8)
            .alias("state"),
        )
        .with_columns(
            (pl.col("state") != pl.col("state").shift(1))
            .fill_null(True)
            .cum_sum()
            .over(["device_serial"])
            .cast(pl.UInt32)
            .alias("group"),
        )
        .with_columns(
            pl.col("sequence_number")
            .cum_count()
            .over(["device_serial", "group"])
            .alias("position_in_group"),
        )
        .explode(["cell_voltages", "cell_position"])
        .with_columns(
            pl.when(
                (pl.col("cell_voltages") >= 3500)
                & (pl.col("state") == 2)
                & (pl.col("position_in_group") >= 10)
            )
            .then(pl.col("delta_charge"))
            .otherwise(None)
            .cast(pl.Int64)
            .forward_fill()
            .over(["device_serial", "cell_position"])
            .alias("q_top"),
        )
        .drop_nulls(subset=["q_top"])
        .with_columns(pl.max_horizontal("q_top", "delta_charge").alias("q_top_ff"))
        .with_columns(
            (
                (capacity_coulombs - (pl.col("q_top_ff") - (pl.col("charge_in") - pl.col("charge_out"))))
                / capacity_coulombs * 100
            ).alias("soc_reference")
        )
    )


def prepare_base(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
    soc_source: SocSource,
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Return long per-cell frame and one frame per cell for SOC/SOH estimation."""
    if soc_source == "per_cell":
        df_base = compute_per_cell_soc(df_raw, capacity_ah, num_cells)
    else:
        df_base = (
            df_raw
            .sort(["device_serial", "sequence_number"])
            .with_columns(
                pl.lit(list(range(num_cells))).cast(pl.List(pl.Int16)).alias("cell_position"),
                (pl.col("cell_voltages") * 1000).cast(pl.List(pl.Int32)),
                pl.col("soc_reported").cast(pl.Float64).alias("soc_reference"),
                ((pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0)
                .over("device_serial", order_by="sequence_number")
                .alias("current_average_raw"),
            )
            .explode(["cell_voltages", "cell_position"])
        )

    df_base = (
        df_base
        .with_columns(
            current=pl.col("current").cast(pl.Float64),
            current_average_raw=pl.col("current_average_raw").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            cell_voltages=pl.col("cell_voltages").cast(pl.Float64),
            soc_reference=pl.col("soc_reference").cast(pl.Float64),
        )
        .with_columns(
            current_actual=(pl.col("current_average_raw") / capacity_ah * LUT_REFERENCE_CAPACITY),
            current_bin=((pl.col("current_average_raw") / capacity_ah * LUT_REFERENCE_CAPACITY) // 5 * 5).cast(pl.Float64),
            cell_temperatures=(pl.col("temperature_battery") // 5 * 5).cast(pl.Float64),
        )
    )

    cell_dfs: list[pl.DataFrame] = []
    for i in range(num_cells):
        cell_dfs.append(
            df_base
            .filter(pl.col("cell_position") == i)
            .filter(pl.col("cell_voltages") > 0)
            .select(
                "device_serial",
                "sequence_number",
                "time_stamp",
                "cell_temperatures",
                "current_average_raw",
                "current_actual",
                "current_bin",
                "cell_voltages",
                "soc_reference",
            )
        )

    return df_base, cell_dfs


# ── SOH processing ────────────────────────────────────────────────────────────
def _apply_soc_update_mode(est: pl.DataFrame, soc_update_mode: SocUpdateMode) -> pl.DataFrame:
    measured_soc = ((pl.col("soc_max") + pl.col("soc_min")) / 2).alias("soc_estimate_raw")
    df = (
        est
        .drop_nulls()
        .drop_nans()
        .sort("sequence_number")
        .with_columns(measured_soc)
        .with_columns(
            soc_estimate_rolling=pl.col("soc_estimate_raw")
            .rolling_mean(window_size=WINDOW_SIZE_SOC, weights=WEIGHTS_SOC, min_samples=WINDOW_SIZE_SOC - 1)
        )
    )

    if soc_update_mode == "ocv_only":
        df = df.with_columns(
            soc_estimate_update=pl.when(pl.col("is_ocv_update"))
            .then(pl.col("soc_estimate_rolling"))
            .otherwise(None)
            .forward_fill()
        )
    else:
        df = df.with_columns(soc_estimate_update=pl.col("soc_estimate_rolling"))

    return df


def process_battery(
    df_raw: pl.DataFrame,
    lut: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
    current_threshold: float,
    resistance_ohm: float,
    soc_source: SocSource,
    soc_update_mode: SocUpdateMode,
) -> pl.DataFrame:
    _, cell_dfs = prepare_base(df_raw, capacity_ah, num_cells, soc_source=soc_source)

    results: list[pl.DataFrame] = []
    for i, df_cell in enumerate(cell_dfs, start=1):
        est = vectorised_soc_estimation(lut, df_cell, resistance_ohm=resistance_ohm)
        part = (
            _apply_soc_update_mode(est, soc_update_mode=soc_update_mode)
            .filter(pl.col("current_actual") <= current_threshold)
            .drop_nulls(subset=["soc_reference", "soc_estimate_update"])
            .drop_nans()
            .with_columns(
                soc_delta=pl.col("soc_reference") - pl.col("soc_estimate_update"),
            )
            .with_columns(
                soh_estimate_raw=100 - pl.col("soc_delta"),
                soh_estimate_nulled=pl.when(pl.col("cell_voltages") < 3280)
                .then(100 - pl.col("soc_delta"))
                .otherwise(None),
            )
            .with_columns(
                soh_estimate_rolling=pl.col("soh_estimate_raw")
                .rolling_mean(window_size=WINDOW_SIZE, weights=WEIGHTS_SOH, min_samples=WINDOW_SIZE - 1),
                soh_estimate_interp=pl.col("soh_estimate_nulled")
                .rolling_mean(window_size=WINDOW_SIZE, min_samples=1)
                .interpolate(method="linear"),
            )
            .with_columns(
                soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
                cell_number=pl.lit(i),
            )
            .filter(pl.col("soh_estimate_ffill") > 0)
        )
        results.append(part)
        print(f"  Cell {i}/{num_cells} done", end="\r", flush=True)

    print()
    if not results:
        return pl.DataFrame()
    return pl.concat(results, how="diagonal_relaxed")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_soh(df_all: pl.DataFrame, serial: str, num_cells: int, capacity_ah: float, args: argparse.Namespace) -> None:
    plot_df = df_all.with_columns(pl.col("cell_number").cast(pl.Utf8).alias("cell"))
    fig = px.line(
        plot_df.to_pandas(),
        x="time_stamp",
        y="soh_estimate_ffill",
        color="cell",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data=[
            "soc_reference",
            "soc_estimate_update",
            "soc_delta",
            "cell_temperatures",
            "current_actual",
            "nearest_current_bin",
            "cell_voltages",
            "voltage_adjusted_mv",
            "is_ocv_update",
            "cell_number",
        ],
        labels={
            "soh_estimate_ffill": "SOH Estimate (%)",
            "time_stamp": "Time",
            "cell": "Cell #",
        },
        title=(
            f"SOH Estimate per cell — {serial} "
            f"({num_cells} cells, {capacity_ah} Ah, source={args.soc_source}, "
            f"updates={args.soc_update_mode}, current ≤ {args.current_threshold} A)"
        ),
    )
    fig.update_traces(opacity=0.8)
    fig.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a battery from Athena and plot per-cell SOH from SOC delta. Use with uv run."
    )
    parser.add_argument("--serial", required=True, help="Device serial number, e.g. P34-000092")
    parser.add_argument("--capacity", required=True, type=float, help="Battery capacity in Ah, e.g. 277")
    parser.add_argument("--num_cells", required=True, type=int, help="Number of cells, e.g. 16 or 32")
    parser.add_argument("--csv_dir", default=".", help="Directory containing constant-current CSV LUT files")
    parser.add_argument("--ocv_csv", default=None, help="OCV CSV path; replaces the near-0 A LUT bin")
    parser.add_argument("--resistance", default=NOMINAL_RESISTANCE, type=float, help="Nominal cell resistance in Ohms")
    parser.add_argument("--soc_source", choices=["reported", "per_cell"], default="reported", help="SOC reference for the delta")
    parser.add_argument("--soc_update_mode", choices=["all", "ocv_only"], default="all", help="Update estimated SOC on all points or only on OCV/idle points")
    parser.add_argument("--current_threshold", default=-100.0, type=float, help="SOH filter on scaled actual current in A (default: -100)")
    parser.add_argument("--start_time", default="2021-01-01", help="Athena lower bound timestamp")
    parser.add_argument("--aws_profile", default="infra_dev", help="AWS profile name")
    parser.add_argument("--output_csv", default=None, help="Optional path to save long-format results CSV")
    args = parser.parse_args()

    print(f"Building LUT from '{args.csv_dir}' …")
    lut = build_lut_from_csvs(args.csv_dir, args.capacity, ocv_csv=args.ocv_csv)
    print(f"Using nominal resistance: {args.resistance} Ω ({args.resistance * 1000:.2f} mΩ)")

    df_raw = fetch_data(
        serial=args.serial,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        aws_profile=args.aws_profile,
        start_time=args.start_time,
    )
    print(f"Fetched {len(df_raw):,} rows for {args.serial}")

    print("Running SOH estimation …")
    df_all = process_battery(
        df_raw=df_raw,
        lut=lut,
        capacity_ah=args.capacity,
        num_cells=args.num_cells,
        current_threshold=args.current_threshold,
        resistance_ohm=args.resistance,
        soc_source=args.soc_source,
        soc_update_mode=args.soc_update_mode,
    )
    print(f"SOH estimation complete — {len(df_all):,} qualifying rows")

    if args.output_csv:
        df_all.write_csv(args.output_csv)
        print(f"Saved results to {args.output_csv}")

    if not df_all.is_empty():
        plot_soh(df_all, args.serial, args.num_cells, args.capacity, args)
    else:
        print("No qualifying rows to plot. Check current_threshold, soc_update_mode, and OCV availability.")


if __name__ == "__main__":
    main()
