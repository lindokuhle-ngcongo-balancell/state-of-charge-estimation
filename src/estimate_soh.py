"""
estimate_soh.py
---------------
Estimate State-of-Health (SOH) per battery from a CSV of telemetry data,
using a voltage-vs-SOH look-up table (LUT) parquet.

Input CSV columns:
    device_serial, sequence_number, time_stamp, current,
    min_cell_voltage, soc, temperature_battery, charge_capacity

Output CSV columns (one row per device_serial):
    device_serial, soh_min, soh_mean, soh_max, soh_std,
    n_windows, last_sequence_number

Usage:
    python estimate_soh.py \
        --input  data/telemetry.csv \
        --lut    data/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
        --output results/soh_estimates.csv \
        [--window 43200]          # rows to use per battery (default 1440*30)
        [--soh    100]            # which SOH slice to pull from LUT
        [--lut-capacity 277]      # nominal Ah of the LUT battery
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 31          # rolling-mean smoothing window
MULTIPLIER_SOH    = 7           # cascade multiplier for higher-order smoothing
LUT_CAPACITY_AH   = 277.0       # default LUT battery nominal capacity
DEFAULT_WINDOW    = 1440 * 30   # default rows per battery for SOH estimation


# ── LUT loader ───────────────────────────────────────────────────────────────
def load_lut(lut_path: str, lut_capacity: float) -> pl.DataFrame:
    """Load and pre-process the voltage-vs-SOH look-up table."""
    df_lut = (
        pl.scan_parquet(lut_path)
        .with_columns(
            (pl.col("q_removed_ah") / lut_capacity * 100)
            .round(0)
            .alias("dod")
        )
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg(
            pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean")
        )
        .sort(["soh", "cell_temperatures", "current_average", "dod"])
        .with_columns(
            pl.col("cell_voltages_mean")
            .diff()
            .over(
                partition_by=["soh", "cell_temperatures", "current_average"],
                order_by="dod",
            )
            .alias("cell_voltages_diff")
        )
        .collect()
    )
    return df_lut


# ── Vectorised SOC estimation ─────────────────────────────────────────────────
def vectorised_soc_estimation(
    df_lut: pl.DataFrame,
    df: pl.DataFrame,
    soh: float = 100,
) -> pl.DataFrame:
    """
    For each row in *df*, bracket the measured voltage against the LUT
    (filtered to *soh*) and return soc_min / soc_max columns.

    df must contain: cell_temperatures, current_average, cell_voltages
    """
    lut = df_lut.filter(pl.col("soh") == soh)
    lut_temp = lut["cell_temperatures"].to_numpy()
    lut_curr = lut["current_average"].to_numpy()
    lut_volt = lut["cell_voltages_mean"].to_numpy()
    lut_dod  = lut["dod"].to_numpy().astype(int)

    temps    = df["cell_temperatures"].to_numpy()
    currents = df["current_average"].to_numpy()
    voltages = df["cell_voltages"].to_numpy()
    n        = len(df)

    soc_min_arr = np.full(n, np.nan)
    soc_max_arr = np.full(n, np.nan)

    unique_keys = np.unique(np.stack([temps, currents], axis=1), axis=0)

    for temp_bin, curr_bin in unique_keys:
        lut_mask = (lut_temp == temp_bin) & (lut_curr == curr_bin)
        if not lut_mask.any():
            continue

        bin_dod  = lut_dod[lut_mask]
        bin_volt = lut_volt[lut_mask]

        valid    = bin_volt >= 0
        bin_dod  = bin_dod[valid]
        bin_volt = bin_volt[valid]
        if len(bin_dod) == 0:
            continue

        sort_idx = np.argsort(bin_dod)
        bin_dod  = bin_dod[sort_idx]
        bin_volt = bin_volt[sort_idx]

        v_max = bin_volt[0]
        v_min = bin_volt[-1]

        data_mask = (temps == temp_bin) & (currents == curr_bin)
        v_query   = voltages[data_mask]

        flipped_volt = bin_volt[::-1]
        flipped_dod  = bin_dod[::-1]

        idx_right = np.searchsorted(flipped_volt, v_query, side="left").clip(
            0, len(flipped_volt) - 1
        )
        idx_left = (idx_right - 1).clip(0, len(flipped_volt) - 1)

        dod_at_left  = flipped_dod[idx_left]
        dod_at_right = flipped_dod[idx_right]

        dod_min_v = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
        dod_max_v = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

        above = v_query > v_max
        below = v_query < v_min

        dod_min_v = np.where(above, 0, np.where(below, 100, dod_min_v))
        dod_max_v = np.where(above, 0, np.where(below, 100, dod_max_v))

        soc_min_arr[data_mask] = (100 - dod_max_v).clip(0, 100)
        soc_max_arr[data_mask] = (100 - dod_min_v).clip(0, 100)

    return df.with_columns(
        [
            pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
            pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
        ]
    )


# ── Exponential weights ───────────────────────────────────────────────────────
def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))
    return (weights / weights.sum()).tolist()


WEIGHTS = make_weights(17, decay=0.7)


# ── Per-battery SOH estimator ─────────────────────────────────────────────────
def estimate_soh_for_battery(
    df_lut: pl.DataFrame,
    df_battery: pl.DataFrame,
    soh_lut: float,
    window_rows: int,
) -> dict:
    """
    Given pre-processed data for a single battery, run the SOC estimation
    pipeline and return summary SOH statistics.
    """
    device_serial        = df_battery["device_serial"][0]
    last_seq             = int(df_battery["sequence_number"].max())
    n_total              = len(df_battery)

    # Use only the last *window_rows* rows
    df_window = df_battery.tail(window_rows)
    df_window = df_window.filter(pl.col("soc_nominal") <= 95)
    if len(df_window) < WINDOW_SIZE + 2:
        return {
            "device_serial":      device_serial,
            "soh_min":            None,
            "soh_mean":           None,
            "soh_max":            None,
            "soh_std":            None,
            "n_points_used":      len(df_window),
            "last_sequence_number": last_seq,
        }

    # ── SOC estimation ────────────────────────────────────────────────────────
    with_soc = vectorised_soc_estimation(df_lut, df_window, soh=soh_lut)

    result = (
        with_soc
        .drop_nulls()
        .drop_nans()
        .with_columns(
            rolling_soc_mean=(
                ((pl.col("soc_max") + pl.col("soc_min")) / 2).rolling_mean(
                    window_size=17, weights=WEIGHTS, min_samples=1
                )
            )
        )
        .filter(pl.col("current_average")<=-100)
        .with_columns(
            soh_estimate=(
                (100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean")))
                .rolling_mean(window_size=1441, min_samples=1)
            )
        )
        # cascade smoothing (mirrors original 2nd / 3rd order passes)
        # .with_columns(
        #     second_order=pl.col("soh_estimate").rolling_mean(
        #         window_size=WINDOW_SIZE * MULTIPLIER_SOH * 3, min_samples=1
        #     )
        # )
        # .with_columns(
        #     third_order=pl.col("second_order").rolling_mean(
        #         window_size=WINDOW_SIZE * MULTIPLIER_SOH * 5, min_samples=1
        #     )
        # )
        # .drop_nulls(subset=["third_order"])
    )

    if result.is_empty():
        vals = None
    else:
        vals = result["soh_estimate"].to_numpy()

    if vals is None or len(vals) == 0:
        return {
            "device_serial":        device_serial,
            "soh_min":              None,
            "soh_mean":             None,
            "soh_max":              None,
            "soh_std":              None,
            "n_points_used":        len(df_window),
            "last_sequence_number": last_seq,
        }

    return {
        "device_serial":        device_serial,
        "soh_min":              float(np.nanmin(vals)),
        "soh_mean":             float(np.nanmean(vals)),
        "soh_max":              float(np.nanmax(vals)),
        "soh_std":              float(np.nanstd(vals)),
        "n_points_used":        len(df_window),
        "last_sequence_number": last_seq,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate per-battery SOH from telemetry CSV + LUT parquet."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input CSV (device_serial, sequence_number, time_stamp, "
             "current, min_cell_voltage, soc, temperature_battery, charge_capacity)",
    )
    parser.add_argument(
        "--lut", "-l", required=True,
        help="Path to voltage-vs-SOH LUT parquet file.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path for output CSV.",
    )
    parser.add_argument(
        "--window", "-w", type=int, default=DEFAULT_WINDOW,
        help=f"Number of most-recent rows per battery to use (default: {DEFAULT_WINDOW}).",
    )
    parser.add_argument(
        "--soh", "-s", type=float, default=100.0,
        help="Which SOH slice to pull from the LUT (default: 100).",
    )
    parser.add_argument(
        "--lut-capacity", type=float, default=LUT_CAPACITY_AH,
        help=f"Nominal Ah capacity of the LUT battery (default: {LUT_CAPACITY_AH}).",
    )
    args = parser.parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    for p in (args.input, args.lut):
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load LUT ──────────────────────────────────────────────────────────────
    print(f"Loading LUT from {args.lut} …")
    df_lut = load_lut(args.lut, args.lut_capacity)
    print(f"  LUT rows: {len(df_lut):,}")

    # ── Load input CSV ────────────────────────────────────────────────────────
    print(f"Loading input CSV from {args.input} …")
    df_raw = pl.read_csv(args.input)

    required_cols = {
        "device_serial", "sequence_number", "current",
        "min_cell_voltage", "soc", "temperature_battery", "charge_capacity",
    }
    missing = required_cols - set(df_raw.columns)
    if missing:
        print(f"ERROR: missing columns in CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    # ── Pre-process ───────────────────────────────────────────────────────────
    # Scale current to the 277 Ah LUT reference by the ratio of capacities.
    # (current_scaled keeps the same C-rate representation as the LUT.)
    df_proc = (
        df_raw
        .sort(["device_serial", "sequence_number"])
        .with_columns(
            # Scale current so that it is expressed in the same units as the LUT
            current_scaled=(
                pl.col("current") / (pl.col("charge_capacity")/3600.0 )* args.lut_capacity
            ),
            # Bin temperature and (scaled) current to nearest 5 — matches LUT binning
            cell_temperatures=(
                (pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)
            ),
        )
        .with_columns(
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
        )
        # Rename columns to match what the estimation functions expect
        .rename(
            {
                "min_cell_voltage": "cell_voltages",
                "soc":              "soc_nominal",
            }
        )
        .with_columns(
            cell_voltages=pl.col("cell_voltages")*1000
        )
        .filter(pl.col("cell_voltages")>0)
        .select(
            "device_serial",
            "sequence_number",
            "cell_temperatures",
            "current_average",
            "cell_voltages",
            "soc_nominal",
            "charge_capacity",
        )
    )

    batteries = df_proc["device_serial"].unique().sort().to_list()
    print(f"  Batteries found: {len(batteries)}")
    print(f"  Window size per battery: {args.window:,} rows")

    # ── Per-battery estimation ────────────────────────────────────────────────
    rows = []
    for i, serial in enumerate(batteries, 1):
        print(f"  [{i}/{len(batteries)}] {serial} …", end=" ", flush=True)
        df_bat = df_proc.filter(pl.col("device_serial") == serial)
        row = estimate_soh_for_battery(df_lut, df_bat, args.soh, args.window)
        rows.append(row)
        if row["soh_mean"] is not None:
            print(
                f"mean={row['soh_mean']:.2f}%  "
                f"[{row['soh_min']:.2f} – {row['soh_max']:.2f}]  "
                f"std={row['soh_std']:.2f}  "
                f"n={row['n_points_used']:,}"
            )
        else:
            print("insufficient data")

    # ── Write output ──────────────────────────────────────────────────────────
    df_out = (
        pl.DataFrame(rows)
        .sort("device_serial")
    )

    df_out.write_csv(args.output)
    print(f"\nOutput written to {args.output}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
