"""
soc_reported_vs_estimate.py
───────────────────────────
Fetch the full history of a single battery from Athena, run the per-cell
SOC estimation and display an animated Plotly chart with:
    • soc_nominal            — one line, reported BMS SOC (same for all cells)
    • rolling_soc_mean_cell_N — one line per cell, LUT-based rolling SOC estimate

Animation is by day (sequence_number // 1440), x-axis is the within-day
sequence offset.

Usage
-----
python soc_reported_vs_estimate.py \
    --serial      BE20000157 \
    --capacity    277 \
    --num_cells   16 \
    --lut_path    data/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
    --lut_soh     100 \
    --aws_profile infra_dev \
    --output_csv  soc_comparison.csv   # optional
"""

import argparse
import boto3
import awswrangler as wr
import polars as pl
import numpy as np
import plotly.express as px

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE_SOC = 7
LUT_REFERENCE_CAPACITY = 277.0


# ── Weights ───────────────────────────────────────────────────────────────────

def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))
    return (weights / weights.sum()).tolist()


WEIGHTS = make_weights(WINDOW_SIZE_SOC, decay=0.7)


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


# ── SOC estimation (vectorised) ───────────────────────────────────────────────

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

        valid = bin_volt >= 0
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

def build_query(serial: str, num_cells: int, capacity_ah: float) -> str:
    capacity_coulombs = capacity_ah * 3600.0
    cell_voltage_cols = "\n".join(
        f"    cell_data.cell_voltage[{i}] AS cell_voltage_{i},"
        for i in range(1, num_cells + 1)
    )
    return f"""
SELECT DISTINCT
    device_serial,
    sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current,
{cell_voltage_cols}
    ( (({capacity_coulombs} - (charge_top - (charge_in - charge_out))) / {capacity_coulombs}) * 100) as soc,
    temperature_battery
FROM analytics.batteries_v3
WHERE device_serial = '{serial}'
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '2023-01-01'
ORDER BY sequence_number ASC
"""


def fetch_data(serial: str, num_cells: int, capacity_ah: float, aws_profile: str) -> pl.DataFrame:
    boto3.setup_default_session(profile_name=aws_profile)
    sql = build_query(serial, num_cells, capacity_ah)
    print(f"Fetching data for {serial} from Athena …")
    pd_df = wr.athena.read_sql_query(
        sql=sql, database="scratch", workgroup="data_engineering"
    )
    return pl.from_pandas(pd_df)


# ── Prepare base DataFrame ────────────────────────────────────────────────────

def prepare_base(df_raw: pl.DataFrame, capacity_ah: float, num_cells: int) -> pl.DataFrame:
    return (
        df_raw
        .sort(["device_serial", "sequence_number"])
        .with_columns(
            soc_nominal=pl.col("soc").cast(pl.Float64),
            current=pl.col("current").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            **{
                f"cell_voltage_{i}": pl.col(f"cell_voltage_{i}").cast(pl.Float64)
                for i in range(1, num_cells + 1)
            },
        )
        .with_columns(
            current_scaled=(pl.col("current") / capacity_ah * LUT_REFERENCE_CAPACITY),
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64))
        )
        .filter(~pl.col("current_average").is_between(-5,0))
    )


# ── Build wide result: one rolling_soc_mean column per cell ──────────────────

def build_wide_result(
    df_base: pl.DataFrame,
    df_lut: pl.DataFrame,
    num_cells: int,
    lut_soh: float,
) -> pl.DataFrame:
    """
    Returns a wide DataFrame indexed by sequence_number with columns:
        sequence_number, sequence_number_original, day,
        soc_nominal,
        soc_cell_1, soc_cell_2, … soc_cell_N
    """
    # Start from the base sequences (use soc_nominal from any cell — it's the same)
    # We'll join each cell's rolling estimate in as a new column.
    seq_base = (
        df_base
        .select("sequence_number", "soc_nominal")
        .unique("sequence_number")
        .sort("sequence_number")
    )

    for i in range(1, num_cells + 1):
        cell_col = f"cell_voltage_{i}"

        df_cell = (
            df_base
            .filter(pl.col(cell_col) > 0)
            .rename({cell_col: "cell_voltages"})
            .with_columns(cell_voltages=pl.col("cell_voltages") * 1000)
            .select(
                "sequence_number", "cell_temperatures", "current_average",
                "cell_voltages", "soc_nominal",
            )
        )

        est = vectorised_soc_estimation(df_lut, df_cell, soh=lut_soh)

        cell_estimate = (
            est
            .drop_nulls()
            .drop_nans()
            .with_columns(
                rolling_soc_mean=(
                    ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=WINDOW_SIZE_SOC, weights=WEIGHTS, min_samples=1)
                ),
            )
            .sort("sequence_number")
            .select(
                "sequence_number",
                pl.col("rolling_soc_mean").alias(f"soc_cell_{i}"),
            )
        )

        seq_base = seq_base.join(cell_estimate, on="sequence_number", how="left")
        print(f"  Cell {i}/{num_cells} done", end="\r", flush=True)

    print()

    # Add day + within-day offset
    return (
        seq_base
        .sort("sequence_number")
        .with_columns(
            day=(pl.col("sequence_number") //( 1_440*7)),
        )
        .with_columns(
            sequence_number_original=pl.col("sequence_number"),
            sequence_number=(
                pl.col("sequence_number") - pl.col("sequence_number").min().over("day")
            ),
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animated day-by-day chart: reported SOC vs per-cell rolling LUT SOC estimate."
    )
    parser.add_argument("--serial",      required=True,  help="Device serial, e.g. BE20000157")
    parser.add_argument("--capacity",    required=True,  type=float, help="Battery capacity in Ah")
    parser.add_argument("--num_cells",   required=True,  type=int,   help="Number of cells")
    parser.add_argument("--lut_path",    required=True,  help="Path to the LUT parquet file")
    parser.add_argument("--lut_soh",     default=100.0,  type=float, help="LUT SOH slice to use (default: 100)")
    parser.add_argument("--aws_profile", default="infra_dev", help="AWS profile name (default: infra_dev)")
    parser.add_argument("--output_csv",  default=None,   help="Optional path to save wide-format CSV")
    args = parser.parse_args()

    print(f"Loading LUT from {args.lut_path} …")
    df_lut = load_lut(args.lut_path)

    df_raw = fetch_data(
        serial=args.serial,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        aws_profile=args.aws_profile,
    )
    print(f"Fetched {len(df_raw):,} rows for {args.serial}")

    df_base = prepare_base(df_raw, args.capacity, args.num_cells)

    print("Running SOC estimation per cell …")
    df_wide = build_wide_result(df_base, df_lut, args.num_cells, args.lut_soh)
    print(f"Wide result: {len(df_wide):,} rows")

    if args.output_csv:
        df_wide.write_csv(args.output_csv)
        print(f"Saved to {args.output_csv}")

    # Unpivot for px.line: soc_nominal + one column per cell → long format
    cell_cols = [f"soc_cell_{i}" for i in range(1, args.num_cells + 1)]
    result_long = df_wide.unpivot(
        on=["soc_nominal"] + cell_cols,
        index=["sequence_number", "sequence_number_original", "day"],
        variable_name="variable",
        value_name="value",
    )

    fig = px.line(
        result_long.to_pandas(),
        x="sequence_number",
        y="value",
        color="variable",
        animation_frame="day",
        hover_data=["sequence_number_original"],
        labels={
            "value": "SOC (%)",
            "variable": "Series",
            "sequence_number": "Sequence (within day)",
            
        },
        title=(
            f"SOC Reported vs Per-Cell LUT Estimate — {args.serial} "
            f"({args.num_cells} cells · {args.capacity} Ah · LUT SOH={args.lut_soh}%)"
        ),
        color_discrete_map={"soc_nominal": "black"},
    )
    fig.update_traces(
        selector=dict(name="soc_nominal"),
        line=dict(width=2.5),
        opacity=1.0,
    )
    fig.update_traces(
        selector=lambda t: t.name != "soc_nominal",
        opacity=0.7,
        line=dict(width=1),
    )
    fig.update_yaxes(range=[-5, 105])
    fig.show()


if __name__ == "__main__":
    main()
