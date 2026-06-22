"""
soh_batch_summary.py
────────────────────
Read a spreadsheet of batteries, fetch all data grouped by (capacity, cell_quantity)
in chunked Athena queries, then run SOH estimation locally in Polars.

Algorithm
---------
Uses the same q_top / charge-counter per-cell SOC algorithm as soh_single_battery.py:
  1. Fetch raw charge_in / charge_out / charge_top + cell_data.cell_voltage (array).
  2. Compute per-cell soc_nominal via the q_top forward-fill approach.
  3. Bin temperature and current, run vectorised_soc_estimation against the LUT.
  4. Derive soh_estimate_ffill per cell, take the last value as the summary SOH.

Strategy
--------
Batteries sharing the same capacity AND cell_quantity are grouped together.
Within each group, serials are fetched and processed chunk_size at a time so
that only one chunk of raw data is ever in memory at once.

Input spreadsheet columns (Excel or CSV):
    device_serial   – e.g. BE20000157
    capacity        – battery capacity in Ah, e.g. 277
    cell_quantity   – number of cells, e.g. 16

Output CSV columns:
    device_serial
    capacity
    cell_quantity
    soh_cell_1 … soh_cell_N   – final forward-filled SOH estimate for each cell
    lowest_soh_per_pack        – minimum SOH across all cells (worst cell)
    time_stamp_of_last_recording

Usage
-----
python soh_batch_summary.py \
    --input              batteries.xlsx \
    --output             soh_summary.csv \
    --lut_path           data/voltage_vs_soh_...parquet \
    --current_threshold  -100   \
    --chunk_size         50     \
    --aws_profile        infra_dev
"""

import argparse
import boto3
import awswrangler as wr
import polars as pl
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE            = 1441
WINDOW_SIZE_SOC        = 7
LUT_REFERENCE_CAPACITY = 277.0   # Ah — capacity the LUT was built for

# ── LUT ───────────────────────────────────────────────────────────────────────

def load_lut(lut_path: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(lut_path)
        .with_columns(
            (pl.col("q_removed_ah") / LUT_REFERENCE_CAPACITY * 100)
            .round(0)
            .alias("dod")
        )
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg(pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean"))
        .sort(["soh", "cell_temperatures", "current_average", "dod"])
        .with_columns(
            pl.col("cell_voltages_mean")
            .diff()
            .over(partition_by=["soh", "cell_temperatures", "current_average"], order_by="dod")
            .alias("cell_voltages_diff")
        )
        .collect()
    )


# ── SOC estimation (vectorised, identical to soh_single_battery.py) ───────────

def vectorised_soc_estimation(
    df_lut: pl.DataFrame, df: pl.DataFrame, soh: float = 100
) -> pl.DataFrame:
    """Vectorised SOC estimation — operates on a single-cell long-format DataFrame."""
    lut      = df_lut.filter(pl.col("soh") == soh)
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
        v_max    = bin_volt[0]
        v_min    = bin_volt[-1]

        data_mask    = (temps == temp_bin) & (currents == curr_bin)
        v_query      = voltages[data_mask]
        flipped_volt = bin_volt[::-1]
        flipped_dod  = bin_dod[::-1]

        idx_right    = np.searchsorted(flipped_volt, v_query, side="left").clip(0, len(flipped_volt) - 1)
        idx_left     = (idx_right - 1).clip(0, len(flipped_volt) - 1)

        dod_at_left  = flipped_dod[idx_left]
        dod_at_right = flipped_dod[idx_right]
        dod_min_v    = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
        dod_max_v    = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

        above     = v_query > v_max
        below     = v_query < v_min
        dod_min_v = np.where(above, 0, np.where(below, 100, dod_min_v))
        dod_max_v = np.where(above, 0, np.where(below, 100, dod_max_v))

        soc_min_arr[data_mask] = (100 - dod_max_v).clip(0, 100)
        soc_max_arr[data_mask] = (100 - dod_min_v).clip(0, 100)

    return df.with_columns([
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
    ])


# ── Athena fetch — one query for a batch of serials ───────────────────────────

def build_group_query(serials: list[str], num_cells: int) -> str:
    """
    Fetch the raw charge-counter columns needed for the per-cell q_top SOC algorithm,
    plus the cell voltage array and temperature — identical schema to soh_single_battery.py.
    """
    serial_list = ", ".join(f"'{s}'" for s in serials)
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
WHERE device_serial IN ({serial_list})
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '2025-01-01' and time_stamp < timestamp '2026-06-01'

"""


def fetch_chunk(
    serials: list[str],
    num_cells: int,
) -> pl.DataFrame:
    """Fetch a single chunk of serials from Athena."""
    sql   = build_group_query(serials, num_cells)
    
    boto3.setup_default_session(profile_name="infra_dev")
    pd_df = wr.athena.read_sql_query(
        sql=sql, database="scratch", workgroup="data_consumers"
    )
    # boto3.client("athena").close()  # free connection pool immediately
    return pl.from_pandas(pd_df) if not pd_df.empty else pl.DataFrame()


# ── Per-cell SOC nominal — q_top / charge-counter algorithm ──────────────────
# (Identical to compute_per_cell_soc in soh_single_battery.py, extended to
#  handle multiple device_serials in one DataFrame via .over("device_serial").)

def compute_per_cell_soc(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """
    Compute per-cell soc_nominal using the q_top / charge-counter algorithm.

    Works on a multi-serial DataFrame; all window operations are partitioned by
    device_serial so each battery's counters are computed independently.

    Returns a long DataFrame with one row per (device_serial, sequence_number,
    cell_position) containing soc_nominal.
    """
    capacity_coulombs = capacity_ah * 3600.0

    return (
        df_raw
        .sort(["device_serial", "sequence_number"])
        # ── charge-counter derived columns (before explode) ───────────────────
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
                (pl.col("cell_voltages") >= 3500)    # mV
                & (pl.col("state") == 2)              # idle
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
        # ── per-cell soc_nominal ──────────────────────────────────────────────
        .with_columns(
            (
                (capacity_coulombs - (pl.col("q_top_ff") - (pl.col("charge_in") - pl.col("charge_out"))))
                / capacity_coulombs * 100
            )
            .alias("soc_nominal"),
        )
    )


# ── Preprocessing — run once per group on the whole block ────────────────────

def preprocess_group(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """
    1. Compute per-cell soc_nominal via the q_top / charge-counter algorithm.
    2. Scale current to LUT reference capacity and bin temperature + current.

    Returns a long DataFrame (one row per device_serial × sequence_number × cell_position)
    ready for vectorised_soc_estimation().
    """
    df_long = compute_per_cell_soc(df_raw, capacity_ah, num_cells)

    return (
        df_long
        .with_columns(
            current=pl.col("current").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            cell_voltages=pl.col("cell_voltages").cast(pl.Float64),
            soc_nominal=pl.col("soc_nominal").cast(pl.Float64),
        )
        .with_columns(
            current_scaled=(pl.col("current") / capacity_ah * LUT_REFERENCE_CAPACITY),
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
        )
    )


# ── Per-battery SOH summary ───────────────────────────────────────────────────

def summarise_battery(
    df_battery: pl.DataFrame,   # preprocessed long-format rows for one device_serial
    df_lut: pl.DataFrame,
    num_cells: int,
    current_threshold: float,
) -> dict:
    """
    Run the per-cell SOH pipeline on a single battery's preprocessed data
    and return a flat summary dict with the last SOH estimate per cell.

    df_battery must already be in long format (one row per sequence × cell_position),
    as produced by preprocess_group().
    """
    soh_per_cell: dict[int, float | None] = {}

    for i in range(num_cells):
        df_cell = (
            df_battery
            .filter(pl.col("cell_position") == i)
            .filter(pl.col("cell_voltages") > 0)
            .select(
                "device_serial", "sequence_number", "cell_temperatures",
                "current_average", "cell_voltages", "soc_nominal", "time_stamp",
            )
        )

        if df_cell.is_empty():
            soh_per_cell[i + 1] = None
            continue

        est = vectorised_soc_estimation(df_lut, df_cell, soh=100)

        df_est = (
            est
            .drop_nulls()
            .drop_nans()
            .with_columns(
                rolling_soc_mean=(
                    ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=WINDOW_SIZE_SOC, min_samples=WINDOW_SIZE_SOC - 1)
                ),
            )
            .filter(pl.col("current_average") <= current_threshold)
            .sort("sequence_number")
            .with_columns(
                soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean"))
            )
            .with_columns(
                soh_estimate_rolling=pl.col("soh_estimate_raw")
                    .rolling_mean(window_size=WINDOW_SIZE, min_samples=1),
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
            .filter(pl.col("soh_estimate_ffill") > 0)
        )

        soh_per_cell[i + 1] = df_est["soh_estimate_ffill"][-1] if not df_est.is_empty() else None

    valid_sohs = [v for v in soh_per_cell.values() if v is not None]
    last_ts    = df_battery.sort("sequence_number")["time_stamp"][-1] if len(df_battery) > 0 else None

    return {
        **{f"soh_cell_{i}": soh_per_cell.get(i) for i in range(1, num_cells + 1)},
        "lowest_soh_per_pack":          min(valid_sohs) if valid_sohs else None,
        "time_stamp_of_last_recording": last_ts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch SOH summary. Batteries with the same capacity + cell_quantity "
            "are grouped together and processed chunk by chunk."
        )
    )
    parser.add_argument("--input",             required=True,  help="Input .xlsx or .csv (columns: device_serial, capacity, cell_quantity)")
    parser.add_argument("--output",            default="soh_summary.csv", help="Output CSV path (default: soh_summary.csv)")
    parser.add_argument("--lut_path",          required=True,  help="Path to LUT parquet file")
    parser.add_argument("--current_threshold", default=-100.0, type=float, help="current_average filter threshold (default: -100)")
    parser.add_argument("--chunk_size",        default=50,     type=int,   help="Max serials per Athena query (default: 10)")
    parser.add_argument("--aws_profile",       default="infra_dev",        help="AWS profile name (default: infra_dev)")
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    if args.input.endswith(".csv"):
        devices_df = pl.read_csv(args.input)
    else:
        devices_df = pl.from_pandas(pd.read_excel(args.input))

    required_cols = {"device_serial", "capacity", "cell_quantity"}
    missing = required_cols - set(devices_df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    devices_df = devices_df.with_columns(
        pl.col("capacity").cast(pl.Float64),
        pl.col("cell_quantity").cast(pl.Int32),
        pl.col("device_serial").cast(pl.Utf8),
    )

  

    print(f"Loading LUT from {args.lut_path} …")
    df_lut = load_lut(args.lut_path)

    # ── Group by (capacity, cell_quantity) ────────────────────────────────────
    groups = (
        devices_df
        .select(["capacity", "cell_quantity"])
        .unique()
        .sort(["capacity", "cell_quantity"])
    )

    summary_rows: list[dict] = []
    max_cells_seen = 0

    for group_row in groups.iter_rows(named=True):
        capacity_ah = float(group_row["capacity"])
        num_cells   = int(group_row["cell_quantity"])
        max_cells_seen = max(max_cells_seen, num_cells)

        serials = (
            devices_df
            .filter(
                (pl.col("capacity") == capacity_ah) &
                (pl.col("cell_quantity") == num_cells)
            )
            ["device_serial"]
            .to_list()
        )

        print(f"\n── Group: capacity={capacity_ah} Ah, cells={num_cells} "
              f"({len(serials)} batteries, chunk_size={args.chunk_size}) ──")

        # ── Process one chunk at a time — raw data is freed after each chunk ──
        for chunk_idx, i in enumerate(range(0, len(serials), args.chunk_size), start=1):
            chunk = serials[i : i + args.chunk_size]
            print(f"  Chunk {chunk_idx}: fetching {len(chunk)} serials from Athena …")

            try:
                df_chunk_raw = fetch_chunk(chunk, num_cells)
            except Exception as exc:
                print(f"  ✗  Athena fetch failed: {exc}")
                for serial in chunk:
                    summary_rows.append({
                        "device_serial": serial, "capacity": capacity_ah, "cell_quantity": num_cells,
                        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
                        "lowest_soh_per_pack": None, "time_stamp_of_last_recording": None,
                    })
                continue

            if df_chunk_raw.is_empty():
                print(f"  ⚠  No data returned")
                for serial in chunk:
                    summary_rows.append({
                        "device_serial": serial, "capacity": capacity_ah, "cell_quantity": num_cells,
                        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
                        "lowest_soh_per_pack": None, "time_stamp_of_last_recording": None,
                    })
                continue

            print(f"  {len(df_chunk_raw):,} rows fetched — preprocessing …")
            df_chunk = preprocess_group(df_chunk_raw, capacity_ah, num_cells)
            del df_chunk_raw  # free raw data immediately

            fetched_serials = set(df_chunk["device_serial"].unique().to_list())

            for serial in chunk:
                if serial not in fetched_serials:
                    print(f"  ⚠  [{serial}] not found in Athena results")
                    summary_rows.append({
                        "device_serial": serial, "capacity": capacity_ah, "cell_quantity": num_cells,
                        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
                        "lowest_soh_per_pack": None, "time_stamp_of_last_recording": None,
                    })
                    continue

                df_battery = df_chunk.filter(pl.col("device_serial") == serial)

                try:
                    soh_row = summarise_battery(df_battery, df_lut, num_cells, args.current_threshold)
                    valid   = soh_row["lowest_soh_per_pack"]
                    print(f"  [{serial}]  rows={len(df_battery):,}  "
                          f"lowest_soh={f'{valid:.1f}%' if valid is not None else 'n/a'}  "
                          f"last_ts={soh_row['time_stamp_of_last_recording']}")
                except Exception as exc:
                    print(f"  ✗  [{serial}] processing error: {exc}")
                    soh_row = {
                        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
                        "lowest_soh_per_pack": None, "time_stamp_of_last_recording": None,
                    }

                summary_rows.append({
                    "device_serial": serial,
                    "capacity":      capacity_ah,
                    "cell_quantity": num_cells,
                    **soh_row,
                })

    # ── Normalise soh_cell columns so all rows have the same width ────────────
    for row in summary_rows:
        for j in range(int(row["cell_quantity"]) + 1, max_cells_seen + 1):
            row.setdefault(f"soh_cell_{j}", None)

    # ── Write output ──────────────────────────────────────────────────────────
    out_df = pl.DataFrame(summary_rows)
    cell_cols    = [f"soh_cell_{j}" for j in range(1, max_cells_seen + 1) if f"soh_cell_{j}" in out_df.columns]
    ordered_cols = (
        ["device_serial", "capacity", "cell_quantity"]
        + cell_cols
        + ["lowest_soh_per_pack", "time_stamp_of_last_recording"]
    )
    out_df = out_df.select(ordered_cols)
    out_df.write_csv(args.output)
    print(f"\n✓  Summary written to {args.output}  ({len(out_df)} batteries)")


if __name__ == "__main__":
    main()
