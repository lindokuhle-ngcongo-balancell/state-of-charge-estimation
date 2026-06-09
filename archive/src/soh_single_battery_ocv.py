"""
soh_single_battery.py
─────────────────────
Fetch the full history of a single battery from Athena, compute a per-cell
SOC nominal (using the q_top / charge-counter algorithm), then run the
per-cell SOC/SOH estimation algorithm and display an interactive Plotly chart.

Usage
-----
python soh_single_battery.py \
    --serial      BE20000157 \
    --capacity    277 \
    --num_cells   16 \
    --csv_dir     data/csv_lut \
    --ocv_csv     data/ocv_data_for_plotting_q.csv \
    --resistance  0.0004 \
    --current_threshold -100          # optional, default -100
    --animate     # optional: animate across SOH 95-105

LUT is built from constant-current CSV files in --csv_dir.
The OCV file (--ocv_csv) is the gold-standard 0 A bin, replacing any
constant-current files that bin to 0 A (e.g. 2 A, 4 A charging).
For currents outside the nearest LUT bin, voltage is IR-corrected:
    V_adj = V_measured + R_mv * (I_actual - I_bin)
No correction is applied when the nearest bin is within ±20 A (near-OCV zone).
"""

import argparse
import os
import re

import boto3
import awswrangler as wr
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 1441
WINDOW_SIZE_SOC  = 7
LUT_REFERENCE_CAPACITY = 277.0
NOMINAL_RESISTANCE     = 0.0004  # Ω — default cell internal resistance

SOH_ANIMATE_MIN  = 95
SOH_ANIMATE_MAX  = 105
SOH_ANIMATE_STEP = 1


# ── Build LUT from CSV files ──────────────────────────────────────────────────

def _parse_csv_filename(fname: str):
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
    Builds the voltage–DOD lookup table from constant-current CSV files and an
    optional OCV file.

    Constant-current CSVs (*Charging_Current_*A.csv / *Discharging_Current_*A.csv):
        - Voltage multiplied by 1000 → mV, rounded to integer.
        - SOC rounded to nearest integer; mean voltage taken per integer SOC.
        - DOD = 100 − SOC.
        - Current signed (discharge negative), scaled to reference capacity,
          then floor-binned to nearest 5 A.
        - Files that bin to 0 A are excluded when an OCV file is supplied.

    OCV file (ocv_csv, optional) — columns: soh, q_removed_ah_ff, cell_voltages.
        - Only SOH=100 rows used.
        - DOD = round(q_removed_ah_ff / capacity_ah * 100, 0).
        - Gold-standard 0 A bin; replaces any constant-current data at that bin.

    Returns a Polars DataFrame: current_bin (float), dod (float),
    cell_voltages_mean (float, mV).  All data at 25 °C, SOH=100.
    """
    rows = []
    for fname in sorted(os.listdir(csv_dir)):
        parsed = _parse_csv_filename(fname)
        if parsed is None:
            continue
        direction, current_abs = parsed
        sign           = 1 if direction == "Charging" else -1
        current_raw    = sign * current_abs
        current_scaled = current_raw / capacity_ah * LUT_REFERENCE_CAPACITY
        current_bin    = float((current_scaled // 5) * 5)

        if current_bin == 0.0 and ocv_csv is not None:
            print(f"  Skipping {fname} (0 A bin — superseded by OCV file)")
            continue

        df = pd.read_csv(os.path.join(csv_dir, fname))
        df["soc_int"]  = df["soc"].round(0).astype(int)
        df["volt_mv"]  = (df["voltage"] * 1000).round(0)
        df_g = df.groupby("soc_int")["volt_mv"].mean().round(0).reset_index()
        df_g["dod"]                = (100 - df_g["soc_int"]).astype(float)
        df_g["current_bin"]        = current_bin
        df_g["cell_voltages_mean"] = df_g["volt_mv"]
        rows.append(df_g[["current_bin", "dod", "cell_voltages_mean"]])

    if not rows and ocv_csv is None:
        raise FileNotFoundError(
            f"No CSV LUT files found in '{csv_dir}'. "
            "Expected filenames containing 'Charging_Current_<N>A' or "
            "'Discharging_Current_<N>A'."
        )

    combined = pd.concat(rows, ignore_index=True)
    combined = (
        combined
        .groupby(["current_bin", "dod"])["cell_voltages_mean"]
        .mean().round(0).reset_index()
    )

    if ocv_csv is not None:
        df_ocv = pd.read_csv(ocv_csv)
        df_ocv = df_ocv[df_ocv["soh"] == 100.0].copy()
        if df_ocv.empty:
            raise ValueError(f"OCV file '{ocv_csv}' contains no rows with soh=100.")
        df_ocv["dod"]     = (df_ocv["q_removed_ah_ff"] / capacity_ah * 100).round(0)
        df_ocv["volt_mv"] = (df_ocv["cell_voltages"] * 1000).round(0)
        df_ocv_g = (
            df_ocv.groupby("dod")["volt_mv"].mean().round(0).reset_index()
            .rename(columns={"volt_mv": "cell_voltages_mean"})
        )
        df_ocv_g["current_bin"] = 0.0
        combined = pd.concat(
            [combined, df_ocv_g[["current_bin", "dod", "cell_voltages_mean"]]],
            ignore_index=True,
        )
        print(f"  OCV file loaded: {len(df_ocv_g)} rows for 0 A bin (SOH=100)")

    lut = pl.from_pandas(combined).sort(["current_bin", "dod"])
    bins = sorted(lut["current_bin"].unique().to_list())
    print(f"LUT ready: {len(lut):,} rows across {len(bins)} current bins: {bins}")
    return lut


# ── SOC estimation (nearest-bin + IR correction) ──────────────────────────────

def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))
    return (weights / weights.sum()).tolist()


WEIGHTS = make_weights(WINDOW_SIZE_SOC, decay=0.9)


def vectorised_soc_estimation(
    df_lut: pl.DataFrame, df: pl.DataFrame, soh: float = 100
) -> pl.DataFrame:
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

        idx_right = np.searchsorted(flipped_volt, v_query, side="left").clip(0, len(flipped_volt) - 1)
        idx_left  = (idx_right - 1).clip(0, len(flipped_volt) - 1)

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

    return df.with_columns([
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
    ])


# ── Athena fetch ──────────────────────────────────────────────────────────────

def build_query(serial: str, num_cells: int) -> str:
    """
    Fetch raw charge-counter columns needed for the per-cell SOC algorithm,
    plus per-cell voltages and temperature.
    Note: we no longer compute soc in SQL — it's done per-cell in Python.
    """
    return f"""
SELECT DISTINCT
    device_serial,
    sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current,
    charge_in,
    charge_out,
    charge_top,
    cell_data.cell_voltage   AS cell_voltages,
    temperature_battery
FROM analytics.batteries_v3
WHERE device_serial = '{serial}'
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '2024-01-01'
ORDER BY sequence_number ASC
"""


def fetch_data(serial: str, num_cells: int, aws_profile: str) -> pl.DataFrame:
    boto3.setup_default_session(profile_name=aws_profile)
    sql = build_query(serial, num_cells)
    print(f"Fetching data for {serial} from Athena …")
    pd_df = wr.athena.read_sql_query(
        sql=sql, database="scratch", workgroup="data_engineering"
    )
    return pl.from_pandas(pd_df)


# ── Per-cell SOC nominal algorithm ────────────────────────────────────────────

def compute_per_cell_soc(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """
    Replicate the q_top / charge-counter SOC algorithm on a per-cell basis.

    Steps (matching the reference snippet):
      1. Sort by sequence_number.
      2. Explode cell_voltages so each row is one (sequence, cell_position).
      3. Compute delta_charge, current_average, state, group, position_in_group.
      4. For each cell: forward-fill q_top from idle rows where voltage >= 3500 mV
         and the pack has been resting for >= 10 samples.
      5. Compute q_removed_ah and soc_nominal per cell from q_top_ff.

    Returns a long DataFrame with one row per (sequence_number, cell_position),
    containing soc_nominal (cell-level).
    """
    capacity_coulombs = capacity_ah * 3600.0

    return (
        df_raw
        .sort(["device_serial", "sequence_number"])
        # ── charge-counter derived columns (row-level, before explode) ────────
        .with_columns(
            pl.lit(list(range(num_cells))).cast(pl.List(pl.Int8)).alias("cell_position"),
            (pl.col("cell_voltages") * 1000).cast(pl.List(pl.Int32)),
            (pl.col("charge_in") - pl.col("charge_out")).alias("delta_charge"),
            # current_average: coulombs per minute → A equivalent
            (
                (pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0
            )
            .over("device_serial")
            .alias("current_average"),
        )
        # ── state: 1=charging, 0=discharging, 2=idle ─────────────────────────
        .with_columns(
            pl.when(pl.col("current_average") > 1).then(1)
              .when(pl.col("current_average") < -1).then(0)
              .otherwise(2)
              .cast(pl.UInt8)
              .alias("state"),
        )
        # ── group: consecutive runs of the same state ─────────────────────────
        .with_columns(
            (pl.col("state") != pl.col("state").shift(1))
            .fill_null(True)
            .cum_sum()
            .over(["device_serial"])
            .cast(pl.UInt32)
            .alias("group"),
        )
        # ── position within each (device, group) run ─────────────────────────
        .with_columns(
            pl.col("sequence_number")
            .cum_count()
            .over(["device_serial", "group"])
            .alias("position_in_group"),
        )
        # ── explode to one row per cell ───────────────────────────────────────
        .explode(["cell_voltages", "cell_position"])
        # ── q_top: latch delta_charge when cell is resting at high voltage ────
        .with_columns(
            pl.when(
                (pl.col("cell_voltages") >= 3500)   # mV — voltages are in mV after explode
                & (pl.col("state") == 2)             # idle
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
        # ── q_top_ff: never let q_top drop below delta_charge ─────────────────
        .with_columns(
            pl.max_horizontal("q_top", "delta_charge").alias("q_top_ff"),
        )
        # ── q_removed_ah ──────────────────────────────────────────────────────
        .with_columns(
            ((pl.col("q_top_ff") - pl.col("delta_charge")) / 3600.0)
            .cast(pl.Float32)
            .alias("q_removed_ah"),
        )
        # ── per-cell soc_nominal ───────────────────────────────────────────────
        .with_columns(
            (
                (capacity_coulombs - (pl.col("q_top_ff") - (pl.col("charge_in") - pl.col("charge_out"))))
                / capacity_coulombs * 100
            )
            .alias("soc_nominal"),
        )
    )


# ── prepare_base: build per-cell DataFrames ready for SOC estimation ─────────

def prepare_base(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """
    1. Compute per-cell soc_nominal via compute_per_cell_soc().
    2. Bin temperature and current for LUT lookup.
    3. Return (df_base_long, cell_dfs) where cell_dfs[i] is the filtered
       DataFrame for cell i+1 ready for vectorised_soc_estimation().
    """
    df_long = compute_per_cell_soc(df_raw, capacity_ah, num_cells)

    df_base = (
        df_long
        .with_columns(
            current=pl.col("current").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            cell_voltages=pl.col("cell_voltages").cast(pl.Float64),
            soc_nominal=pl.col("soc_nominal").cast(pl.Float64),
        )
        .with_columns(
            # Scale current to LUT reference capacity
            current_scaled=(pl.col("current") / capacity_ah * LUT_REFERENCE_CAPACITY),
            # Bin temperature to nearest 5 °C
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(
            # Bin current to nearest 5 A
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
        )
        # .filter(~pl.col("current_average").is_between(-5, 0, closed="both"))
    )

    # One DataFrame per cell (0-indexed cell_position → cells 0..N-1)
    cell_dfs = []
    for i in range(num_cells):
        cell_dfs.append(
            df_base
            .filter(pl.col("cell_position") == i)
            .filter(pl.col("cell_voltages") > 0)
            # Voltages are already in mV from the explode; keep consistent
            .select(
                "device_serial", "sequence_number", "cell_temperatures",
                "current_average", "cell_voltages", "soc_nominal", "time_stamp",
            )
        )

    return df_base, cell_dfs


# ── process_battery ───────────────────────────────────────────────────────────

def process_battery(
    df_raw: pl.DataFrame,
    df_lut: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
    current_threshold: float,
) -> pl.DataFrame:
    """Run SOH estimation for every cell and return a long-format DataFrame."""
    _, cell_dfs = prepare_base(df_raw, capacity_ah, num_cells)

    results = []
    for i, df_cell in enumerate(cell_dfs, start=1):
        est = vectorised_soc_estimation(df_lut, df_cell, soh=100)
        results.append(
            est
            .drop_nulls()
            .drop_nans()
            .with_columns(
                rolling_soc_mean=(
                    ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=WINDOW_SIZE_SOC, weights=make_weights(WINDOW_SIZE_SOC, decay=0.9),  min_samples=WINDOW_SIZE_SOC-1)
                ),
            )
            .filter(pl.col("current_average")<= current_threshold)
            .drop_nulls()
            .drop_nans()
            .sort("sequence_number")
            .with_columns(
                soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
            )
            .with_columns(
                soh_estimate_rolling=pl.col("soh_estimate_raw")
                    .rolling_mean(window_size=WINDOW_SIZE,  weights=make_weights(WINDOW_SIZE, decay=0.7),min_samples=WINDOW_SIZE-1),
                soh_estimate_nulled=pl.when(pl.col("cell_voltages") < 3280)
                    .then(pl.col("soh_estimate_raw"))
                    .otherwise(None),
            )
            .with_columns(
                soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
                soh_estimate_interp=pl.col("soh_estimate_nulled")
                    .rolling_mean(window_size=WINDOW_SIZE, min_samples=1)
                    .interpolate(method="linear"),
            )
            .with_columns(pl.lit(i).alias("cell_number"))
            .filter(pl.col("soh_estimate_ffill") > 0)
        )

    return pl.concat(results, how="diagonal_relaxed")


# ── animate_soh_across_range ──────────────────────────────────────────────────

def animate_soh_across_range(
    df_raw: pl.DataFrame,
    df_lut: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
    current_threshold: float,
    serial: str,
) -> go.Figure:
    soh_values = list(range(SOH_ANIMATE_MIN, SOH_ANIMATE_MAX + 1, SOH_ANIMATE_STEP))
    colors = px.colors.qualitative.Dark24

    _, cell_dfs = prepare_base(df_raw, capacity_ah, num_cells)

    print(f"Building animation frames for SOH {SOH_ANIMATE_MIN}–{SOH_ANIMATE_MAX} …")
    all_frame_data: dict[int, pl.DataFrame] = {}

    for soh_val in soh_values:
        print(f"  SOH={soh_val}", end="\r", flush=True)
        frame_parts = []
        for i, df_cell in enumerate(cell_dfs, start=1):
            est = vectorised_soc_estimation(df_lut, df_cell, soh=soh_val)
            part = (
                est
                .drop_nulls()
                .drop_nans()
                .with_columns(
                    rolling_soc_mean=(
                        ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                        .rolling_mean(window_size=WINDOW_SIZE_SOC, weights=WEIGHTS, min_samples=1)
                    ),
                )
                .filter(pl.col("cell_voltages") <= 3270)
                .sort("sequence_number")
                .with_columns(
                    soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
                )
                .with_columns(
                    soh_estimate_rolling=pl.col("soh_estimate_raw")
                        .rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
                )
                .with_columns(
                    soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill(),
                )
                .filter(pl.col("soh_estimate_ffill") > 0)
                .select("time_stamp", "soh_estimate_ffill")
                .with_columns(pl.lit(i).cast(pl.Utf8).alias("cell"))
            )
            frame_parts.append(part)

        all_frame_data[soh_val] = pl.concat(frame_parts, how="diagonal_relaxed")

    print("\nAssembling Plotly figure …")

    all_estimates = pl.concat(list(all_frame_data.values()))["soh_estimate_ffill"]
    y_min = float(all_estimates.min()) - 2
    y_max = float(all_estimates.max()) + 2

    cell_labels = [str(i) for i in range(1, num_cells + 1)]
    first_df = all_frame_data[soh_values[0]].to_pandas()
    initial_traces = []
    for idx, cell_label in enumerate(cell_labels):
        sub = first_df[first_df["cell"] == cell_label]
        initial_traces.append(
            go.Scatter(
                x=sub["time_stamp"],
                y=sub["soh_estimate_ffill"],
                mode="lines",
                name=f"Cell {cell_label}",
                line=dict(color=colors[idx % len(colors)]),
                opacity=0.85,
            )
        )

    frames = []
    for soh_val in soh_values:
        pdf = all_frame_data[soh_val].to_pandas()
        frame_traces = []
        for cell_label in cell_labels:
            sub = pdf[pdf["cell"] == cell_label]
            frame_traces.append(go.Scatter(x=sub["time_stamp"], y=sub["soh_estimate_ffill"]))
        frames.append(go.Frame(data=frame_traces, name=str(soh_val)))

    sliders = [dict(
        active=0,
        currentvalue=dict(prefix="LUT SOH: ", font=dict(size=14)),
        pad=dict(t=50),
        steps=[
            dict(
                method="animate",
                args=[[str(soh_val)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(soh_val),
            )
            for soh_val in soh_values
        ],
    )]

    updatemenus = [dict(
        type="buttons", showactive=False, y=1.15, x=0.0, xanchor="left",
        buttons=[
            dict(label="▶  Play",  method="animate",
                 args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True, transition=dict(duration=150))]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
        ],
    )]

    fig = go.Figure(
        data=initial_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=(
                    f"SOH Estimate per cell — {serial} ({num_cells} cells, {capacity_ah} Ah)<br>"
                    f"<sup>Animated across LUT SOH {SOH_ANIMATE_MIN}–{SOH_ANIMATE_MAX}. "
                    f"Use the slider or Play button.</sup>"
                ),
                font=dict(size=16),
            ),
            xaxis=dict(title="Time"),
            yaxis=dict(title="SOH Estimate (%)", range=[y_min, y_max]),
            legend=dict(title="Cell #"),
            updatemenus=updatemenus,
            sliders=sliders,
            hovermode="x unified",
        ),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch a battery's full history from Athena and plot per-cell SOH."
    )
    parser.add_argument("--serial",            required=True,  help="Device serial number, e.g. BE20000157")
    parser.add_argument("--capacity",          required=True,  type=float, help="Battery capacity in Ah, e.g. 277")
    parser.add_argument("--num_cells",         required=True,  type=int,   help="Number of cells, e.g. 16")
    parser.add_argument("--current_threshold", default=-100.0, type=float, help="current_average filter threshold (default: -100)")
    parser.add_argument("--lut_path",          required=True,  help="Path to the LUT parquet file")
    parser.add_argument("--aws_profile",       default="infra_dev", help="AWS profile name (default: infra_dev)")
    parser.add_argument("--output_csv",        default=None,   help="Optional path to save results CSV")
    parser.add_argument(
        "--animate", action="store_true",
        help=f"Animate across LUT SOH {SOH_ANIMATE_MIN}–{SOH_ANIMATE_MAX} instead of static plot.",
    )
    args = parser.parse_args()

    print(f"Loading LUT from {args.lut_path} …")
    df_lut = load_lut(args.lut_path)

    df_raw = fetch_data(
        serial=args.serial,
        num_cells=args.num_cells,
        aws_profile=args.aws_profile,
    )
    print(f"Fetched {len(df_raw):,} rows for {args.serial}")

    if args.animate:
        fig = animate_soh_across_range(
            df_raw=df_raw,
            df_lut=df_lut,
            capacity_ah=args.capacity,
            num_cells=args.num_cells,
            current_threshold=args.current_threshold,
            serial=args.serial,
        )
        fig.show()
        return

    print("Running SOH estimation …")
    df_all = process_battery(
        df_raw=df_raw,
        df_lut=df_lut,
        capacity_ah=args.capacity,
        num_cells=args.num_cells,
        current_threshold=args.current_threshold,
    )
    print(f"SOH estimation complete — {len(df_all):,} qualifying rows")

    if args.output_csv:
        df_all.write_csv(args.output_csv)
        print(f"Saved results to {args.output_csv}")

    plot_df = df_all.with_columns(pl.col("cell_number").cast(pl.Utf8).alias("cell"))

    fig = px.line(
        plot_df.to_pandas(),
        x="time_stamp",
        y="soh_estimate_ffill",
        color="cell",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data=["cell_temperatures", "current_average", "cell_voltages", "cell_number"],
        labels={
            "soh_estimate_ffill": "SOH Estimate (%)",
            "time_stamp": "Time",
            "cell": "Cell #",
        },
        title=(
            f"SOH Estimate per cell — {args.serial} "
            f"({args.num_cells} cells, {args.capacity} Ah, "
            f"current threshold ≤ {args.current_threshold} A)"
        ),
    )
    fig.update_traces(opacity=0.8)
    fig.show()


if __name__ == "__main__":
    main()