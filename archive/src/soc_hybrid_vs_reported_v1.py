
"""
soc_hybrid_vs_reported.py
─────────────────────────
Fetches data from Athena and compares three SoC series:
    soc_nominal   — BMS-reported
    soc_rolling   — rolling LUT mean (original algo)
    soc_hybrid    — LUT reset at high current + Coulomb counting between

Memory strategy
───────────────
Stage 1  Process one cell at a time: LUT estimation → write to tmp parquet → discard.
Stage 2  Apply hybrid per cell using precomputed pack-level mode schedule.
         Accumulate means into O(n) numpy arrays, never widen the DataFrame.

Usage
-----
uv run soc_hybrid_vs_reported.py \\
    --serial      BE20000157 \\
    --capacity    277         \\
    --num_cells   32          \\
    --lut_path    data/lut.parquet \\
    --start_date  2026-05-01  \\
    --end_date    2026-05-25  \\
    --aws_profile infra_dev   \\
    --output_html comparison.html \\
    --stats_html  stats.html
"""
from __future__ import annotations

import argparse
import gc
import logging
import logging.handlers
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import awswrangler as wr
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.handlers.RotatingFileHandler(
            "soc_hybrid_vs_reported.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
LOG = logging.getLogger(__name__)

WINDOW_SIZE_SOC = 13
LUT_REFERENCE_CAPACITY = 277.0


# ── LUT ───────────────────────────────────────────────────────────────────────
def load_lut(lut_path: str) -> pl.DataFrame:
    return (
    pl.scan_parquet(
        r"data\\voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet",
    )
    .with_columns((pl.col("q_removed_ah") / (277) * 100).round(0).alias("dod"))
    .filter(pl.col("dod").is_between(0, 100, closed="both"))
    .group_by(["soh", "cell_temperatures", "current_average", "dod"])
    .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
    .sort(["soh", "cell_temperatures", "current_average", "dod"])
    .with_columns(
        pl.col("cell_voltages_mean")
        .diff()
        .over(
            partition_by=["soh", "cell_temperatures", "current_average"], order_by="dod"
        )
        .alias("cell_voltages_diff")
    )
    .filter(pl.col("soh").is_between(98, 102, closed="both"))
    .group_by(["cell_temperatures", "current_average", "dod"])
    .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
    .with_columns(
        pl.col("dod")
        .count()
        .over(["cell_temperatures", "current_average"])
        .alias("dod_count")
    )
    .filter(pl.col("dod_count") >= 85)
    # .with_columns(
    #     pl.when((pl.col('cell_voltages_diff').abs() >3.5) | (pl.col("dod")==0)).then(pl.col('cell_voltages_mean')).otherwise(-1).alias('cell_voltages_mean')
    # )
    .with_columns(
        pl.col("cell_voltages_mean")
        .diff()
        .over(
            partition_by=[
                "current_average",
                "cell_temperatures",
            ],
            order_by="dod",
            descending=True,
        )
        .alias("cell_voltages_diff"),
        pl.col("dod")
        .diff()
        .over(
            partition_by=[
                "current_average",
                "cell_temperatures",
            ],
            order_by="dod",
            descending=False,
        )
        .alias("dod_diff"),
    )
    .filter(pl.col("dod_diff").abs() == 1)
    .filter(pl.col("cell_voltages_diff").abs()>=10)
    .collect()
)


# ── Vectorised SOC estimation ─────────────────────────────────────────────────
def vectorised_soc_estimation(
    df_lut: pl.DataFrame,
    df: pl.DataFrame,
    resistance_ohm: float = 0.00025,
    soh: float = 100,
) -> pl.DataFrame:
    lut_temp = df_lut["cell_temperatures"].to_numpy()
    lut_curr = df_lut["current_average"].to_numpy()
    lut_volt = df_lut["cell_voltages_mean"].to_numpy()
    lut_dod  = df_lut["dod"].to_numpy()

    temps_actual    = df["cell_temperatures"].to_numpy()
    currents_actual = df["current_average"].to_numpy()
    voltages_actual = df["cell_voltages"].to_numpy()
    n = len(df)

    soc_min_arr      = np.full(n, np.nan)
    soc_max_arr      = np.full(n, np.nan)
    matched_temp_arr = np.full(n, np.nan)
    matched_curr_arr = np.full(n, np.nan)
    voltage_adj_arr  = np.full(n, np.nan)
    r_mohm           = resistance_ohm * 1000.0
    unique_temps     = np.unique(lut_temp)

    for i in range(n):
        t = temps_actual[i]
        matched_temp_arr[i] = (
            np.median(unique_temps) if np.isnan(t)
            else unique_temps[np.argmin(np.abs(unique_temps - t))]
        )
    for temp_bin in np.unique(matched_temp_arr[~np.isnan(matched_temp_arr)]):
        tmask = lut_temp == temp_bin
        if not tmask.any():
            continue
        avail = np.unique(lut_curr[tmask])
        for i in np.where(matched_temp_arr == temp_bin)[0]:
            ia = currents_actual[i]
            cb = avail[np.argmin(np.abs(avail))] if np.isnan(ia) else avail[np.argmin(np.abs(avail - ia))]
            matched_curr_arr[i] = cb
            voltage_adj_arr[i]  = voltages_actual[i] 
    for temp_bin, curr_bin in np.unique(
        np.stack([matched_temp_arr, matched_curr_arr], axis=1), axis=0
    ):
        if np.isnan(temp_bin) or np.isnan(curr_bin):
            continue
        lm = (lut_temp == temp_bin) & (lut_curr == curr_bin)
        if not lm.any():
            continue
        bd, bv = lut_dod[lm], lut_volt[lm]
        valid  = np.isfinite(bv) & np.isfinite(bd) & (bv >= 0)
        bd, bv = bd[valid], bv[valid]
        if len(bd) == 0:
            continue
        idx    = np.argsort(bd);  bd, bv = bd[idx], bv[idx]
        v_max, v_min = bv[0], bv[-1]
        dm     = (matched_temp_arr == temp_bin) & (matched_curr_arr == curr_bin)
        vq     = voltage_adj_arr[dm]
        fv, fd = bv[::-1], bd[::-1]
        ir     = np.searchsorted(fv, vq, side="left").clip(0, len(fv) - 1)
        il     = (ir - 1).clip(0, len(fv) - 1)
        dlo    = np.minimum(fd[il], fd[ir]).clip(0, 100)
        dhi    = np.maximum(fd[il], fd[ir]).clip(0, 100)
        above, below = vq > v_max, vq < v_min
        dlo = np.where(above, 0, np.where(below, 100, dlo))
        dhi = np.where(above, 0, np.where(below, 100, dhi))
        soc_min_arr[dm] = (100 - dhi).clip(0, 100)
        soc_max_arr[dm] = (100 - dlo).clip(0, 100)

    return df.with_columns([
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
    ])


# ── Athena ────────────────────────────────────────────────────────────────────
def build_query(serial, num_cells, capacity_ah, start_date, end_date):
    cc = capacity_ah * 3600.0
    cols = "\n".join(
        f"    cell_data.cell_voltage[{i}] AS cell_voltage_{i},"
        for i in range(1, num_cells + 1)
    )
    return f"""
SELECT DISTINCT
    device_serial, sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current, charge_in, charge_out,
{cols}
    ((({cc} - (charge_top - (charge_in - charge_out))) / {cc}) * 100) as soc,
    temperature_battery
FROM analytics.batteries_v3
WHERE device_serial = '{serial}'
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '{start_date}'
  AND time_stamp <  timestamp '{end_date}'
ORDER BY sequence_number ASC
"""


def fetch_data(serial, num_cells, capacity_ah, start_date, end_date, aws_profile):
    boto3.setup_default_session(profile_name=aws_profile)
    LOG.info("Fetching %s  %s → %s …", serial, start_date, end_date)
    return pl.from_pandas(
        wr.athena.read_sql_query(
            sql=build_query(serial, num_cells, capacity_ah, start_date, end_date),
            database="scratch", workgroup="data_engineering",
        )
    )


# ── Prepare base ──────────────────────────────────────────────────────────────
def prepare_base(df_raw: pl.DataFrame, capacity_ah: float, num_cells: int) -> pl.DataFrame:
    return (
        df_raw.sort(["device_serial", "sequence_number"])
        .with_columns(
            soc_nominal=pl.col("soc").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            **{f"cell_voltage_{i}": pl.col(f"cell_voltage_{i}").cast(pl.Float64)
               for i in range(1, num_cells + 1)},
        )
        .with_columns(
            # Net coulombs into battery per 60 s window.
            # Positive = charging (SoC rises), negative = discharging.
            delta_charge=(
                (pl.col("charge_in").diff() - pl.col("charge_out").diff())
                .over("device_serial", order_by="sequence_number")
                .cast(pl.Float64)
            ),
            current_scaled=(pl.col("current") / capacity_ah * LUT_REFERENCE_CAPACITY),
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(
            current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64)),
        )
    )


# ── Mode schedule (LUT-membership based) ──────────────────────────────────────
def compute_mode_schedule(
    temps: list,
    currents: list,
    valid_pairs: set[tuple[float, float]],
) -> tuple[list[str], list[int]]:
    """
    Walk the time series once and decide UNINIT / LUT / CC for every row.

    A row is a LUT reset iff its (cell_temperatures, current_average) pair
    exists in `valid_pairs` — i.e. survived the LUT filtering pipeline. The
    filtered LUT is the single source of truth for what counts as a valid
    operating point; no current-threshold knobs are needed.

    Returns (modes, reset_ids) — both length n.
    """
    modes: list[str] = []
    reset_ids: list[int] = []
    reset_count = 0
    initialized = False

    for t, c in zip(temps, currents):
        if t is None or c is None:
            is_reset = False
        else:
            is_reset = (float(t), float(c)) in valid_pairs
        if is_reset:
            reset_count += 1
            initialized = True
            mode = "LUT"
        elif not initialized:
            mode = "UNINIT"
        else:
            mode = "CC"
        modes.append(mode)
        reset_ids.append(reset_count)

    return modes, reset_ids


# ── Per-cell hybrid (called once per cell, tiny memory) ───────────────────────
def apply_hybrid_single_cell(
    modes: list[str],
    lut_mids: list,       # per-row LUT midpoint for this cell (None if no LUT data)
    delta_charges: list,  # net coulombs per 60 s (shared across cells)
    capacity_ah: float,
) -> list:
    """
    Apply the precomputed mode schedule to one cell's LUT midpoints.
    Returns list of hybrid SoC values (None until first LUT reset).
    """
    soc_hybrid: list = []
    prev_soc: float | None = None

    for mode, mid, dc in zip(modes, lut_mids, delta_charges):
        dc = dc if dc is not None else 0.0
        d_soc = dc / (capacity_ah * 3600.0) * 100.0   # exact — direct from coulombs

        if mode == "LUT":
            if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
                prev_soc = float(mid)
            soc_hybrid.append(prev_soc)
        elif mode == "CC":
            if prev_soc is not None:
                prev_soc = float(np.clip(prev_soc + d_soc, 0.0, 100.0))
            soc_hybrid.append(prev_soc)
        else:   # UNINIT
            soc_hybrid.append(None)

    return soc_hybrid


# ── Main build loop ───────────────────────────────────────────────────────────
def build_wide_result(
    df_base: pl.DataFrame,
    df_lut: pl.DataFrame,
    num_cells: int,
    lut_soh: float,
    capacity_ah: float,
    tmp_dir: str | None = None,
) -> pl.DataFrame:
    """
    Processes one cell at a time to stay within RAM.

    For each cell
        1. Run vectorised_soc_estimation → soc_rolling, lut_mid
        2. Write {sequence_number, soc_cell_i, lut_mid_i} to tmp parquet
        3. Apply hybrid state machine for that cell only
        4. Accumulate rolling/hybrid into O(n) numpy sum arrays
        5. Discard per-cell data and call gc.collect()

    Final DataFrame never holds all 32 cells at once.
    """
    tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    LOG.info("Temp cell files → %s", tmp)

    # ── Narrow base (kept in memory throughout) ───────────────────────────
    seq_base = (
        df_base
        .select("sequence_number", "time_stamp", "current", "current_average",
                "cell_temperatures", "soc_nominal", "delta_charge")
        .unique("sequence_number")
        .sort("sequence_number")
    )
    n        = seq_base.height
    seq_list = seq_base["sequence_number"].to_list()

    # ── Valid (temp, current) pairs from filtered LUT ─────────────────────
    # The LUT pipeline has already filtered to genuinely valid operating
    # points; any (cell_temperatures, current_average) pair present here
    # is a legitimate LUT reset point.
    valid_pairs: set[tuple[float, float]] = {
        (float(t), float(c))
        for t, c in zip(
            df_lut["cell_temperatures"].to_list(),
            df_lut["current_average"].to_list(),
        )
    }
    LOG.info("LUT defines %d valid (temp, current) reset pairs", len(valid_pairs))

    # ── Mode schedule — per-cell-callable, but inputs are pack-level so   ─
    #    we compute once and reuse across cells (cell_temperatures is      ─
    #    derived from pack temperature_battery in prepare_base).            ─
    modes, reset_ids = compute_mode_schedule(
        seq_base["cell_temperatures"].to_list(),
        seq_base["current_average"].to_list(),
        valid_pairs,
    )
    delta_charges = seq_base["delta_charge"].to_list()
    LOG.info("Mode schedule: UNINIT=%d  LUT=%d  CC=%d",
             modes.count("UNINIT"), modes.count("LUT"), modes.count("CC"))

    # ── Running mean accumulators (O(n) floats) ───────────────────────────
    rolling_sum = np.zeros(n, dtype=np.float64)
    rolling_cnt = np.zeros(n, dtype=np.float64)
    hybrid_sum  = np.zeros(n, dtype=np.float64)
    hybrid_cnt  = np.zeros(n, dtype=np.float64)

    # ── Per-cell loop ─────────────────────────────────────────────────────
    for i in range(1, num_cells + 1):
        cell_col = f"cell_voltage_{i}"

        df_cell = (
            df_base
            .filter(pl.col(cell_col) > 0)
            .rename({cell_col: "cell_voltages"})
            .with_columns(cell_voltages=pl.col("cell_voltages") * 1000)
            .select("sequence_number", "cell_temperatures", "current_average",
                    "cell_voltages", "soc_nominal")
        )

        est = vectorised_soc_estimation(df_lut, df_cell, soh=lut_soh)

        cell_r = (
            est.drop_nulls().drop_nans().sort("sequence_number")
            .with_columns(
                soc_rolling=(
                    ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=WINDOW_SIZE_SOC, min_samples=WINDOW_SIZE_SOC - 1)
                ),
                lut_mid=((pl.col("soc_max") + pl.col("soc_min")) / 2),
            )
            .select("sequence_number", "soc_rolling", "lut_mid")
        )

        # Write to tmp so caller can inspect individual cells if needed
        cell_r.write_parquet(tmp / f"cell_{i:03d}.parquet")

        # Build aligned lists (index = position in seq_base)
        r_map   = dict(zip(cell_r["sequence_number"].to_list(), cell_r["soc_rolling"].to_list()))
        mid_map = dict(zip(cell_r["sequence_number"].to_list(), cell_r["lut_mid"].to_list()))
        rolling_vals = [r_map.get(s)   for s in seq_list]
        lut_mids     = [mid_map.get(s) for s in seq_list]

        # Hybrid for this cell
        hybrid_vals = apply_hybrid_single_cell(modes, lut_mids, delta_charges, capacity_ah)

        # Accumulate into running means
        for idx in range(n):
            rv = rolling_vals[idx]
            hv = hybrid_vals[idx]
            if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                rolling_sum[idx] += rv
                rolling_cnt[idx] += 1
            if hv is not None:
                hybrid_sum[idx] += hv
                hybrid_cnt[idx] += 1

        # Free everything and force GC before next cell
        del est, df_cell, cell_r, rolling_vals, lut_mids, hybrid_vals, r_map, mid_map
        gc.collect()
        print(f"  Cell {i}/{num_cells}", end="\r", flush=True)

    print()

    mean_rolling = np.where(rolling_cnt > 0, rolling_sum / rolling_cnt, np.nan)
    mean_hybrid  = np.where(hybrid_cnt  > 0, hybrid_sum  / hybrid_cnt,  np.nan)

    result = seq_base.with_columns([
        pl.Series("soc_mode",        modes,        dtype=pl.Utf8),
        pl.Series("reset_id",        reset_ids,    dtype=pl.Int32),
        pl.Series("mean_soc_rolling", mean_rolling, dtype=pl.Float64),
        pl.Series("mean_soc_hybrid",  mean_hybrid,  dtype=pl.Float64),
    ])

    return (
        result.sort("sequence_number")
        .with_columns(day=(pl.col("sequence_number") // (1_440 * 7)))
        .with_columns(
            sequence_number_original=pl.col("sequence_number"),
            sequence_number=(
                pl.col("sequence_number") - pl.col("sequence_number").min().over("day")
            ),
        )
    )


# ── Tracking statistics ───────────────────────────────────────────────────────
def compute_tracking_stats(df: pl.DataFrame) -> dict:
    valid = (
        df.filter(pl.col("mean_soc_hybrid").is_not_null() & pl.col("soc_mode").ne("UNINIT"))
        .with_columns(soc_nom=pl.col("soc_nominal").cast(pl.Float64))
        .with_columns(
            err_hybrid =(pl.col("mean_soc_hybrid")  - pl.col("soc_nom")),
            err_rolling=(pl.col("mean_soc_rolling") - pl.col("soc_nom")),
        )
    )

    def _s(series: pl.Series) -> dict:
        s = series.drop_nulls().drop_nans()
        if s.len() == 0:
            return {}
        return {
            "n":    s.len(),
            "bias": round(float(s.mean()), 3),
            "mae":  round(float(s.abs().mean()), 3),
            "rmse": round(float((s ** 2).mean() ** 0.5), 3),
            "p90":  round(float(s.abs().quantile(0.90)), 3),
            "max":  round(float(s.abs().max()), 3),
        }

    overall = {"hybrid": _s(valid["err_hybrid"]), "rolling": _s(valid["err_rolling"])}
    if overall["rolling"].get("mae") and overall["hybrid"].get("mae"):
        r = (overall["rolling"]["mae"] - overall["hybrid"]["mae"]) / overall["rolling"]["mae"] * 100
        overall["mae_reduction_pct"] = round(r, 1)

    modes = {r["soc_mode"]: r["count"]
             for r in df["soc_mode"].value_counts().iter_rows(named=True)}
    n_days = df["day"].n_unique() if "day" in df.columns else 1
    modes["resets_per_day"] = round(df.filter(pl.col("soc_mode") == "LUT").height / max(n_days, 1), 1)

    by_day = []
    if "day" in valid.columns:
        for day in sorted(valid["day"].unique().to_list()):
            sub = valid.filter(pl.col("day") == day)
            n_resets = df.filter((pl.col("day") == day) & (pl.col("soc_mode") == "LUT")).height
            by_day.append({
                "day": day, "n": sub.height, "n_resets": n_resets,
                "hybrid_bias":  round(float(sub["err_hybrid"].mean()), 2),
                "hybrid_mae":   round(float(sub["err_hybrid"].abs().mean()), 2),
                "rolling_bias": round(float(sub["err_rolling"].drop_nulls().mean()), 2),
                "rolling_mae":  round(float(sub["err_rolling"].drop_nulls().abs().mean()), 2),
            })

    by_soc = (
        valid
        .with_columns(soc_bucket=((pl.col("soc_nom") / 10).floor() * 10))
        .group_by("soc_bucket")
        .agg([
            pl.col("err_hybrid").abs().mean().round(3).alias("hybrid_mae"),
            pl.col("err_rolling").abs().mean().round(3).alias("rolling_mae"),
            pl.len().alias("n"),
        ])
        .sort("soc_bucket")
        .to_dicts()
    )

    # CC drift: error vs minutes since last reset (reveals capacity mismatch)
    cc_rows = valid.filter(pl.col("soc_mode") == "CC")
    cc_drift = []
    if "reset_id" in cc_rows.columns and cc_rows.height > 0:
        cc_drift = (
            cc_rows
            .with_columns(steps=pl.int_range(pl.len()).over("reset_id"))
            .group_by("steps")
            .agg([
                pl.col("err_hybrid").abs().mean().round(3).alias("mean_abs_err"),
                pl.col("err_hybrid").mean().round(3).alias("mean_err"),
                pl.len().alias("n"),
            ])
            .sort("steps")
            .filter(pl.col("n") >= 5)
            .to_dicts()
        )

    reset_quality = _s(valid.filter(pl.col("soc_mode") == "LUT")["err_hybrid"])

    return {
        "overall": overall, "modes": modes, "by_day": by_day,
        "by_soc": by_soc, "cc_drift": cc_drift, "reset_quality": reset_quality,
    }


def print_tracking_stats(s: dict) -> None:
    ov = s["overall"]
    print("\n══════════════════════════════════════════════════════")
    print("  SoC Tracking Statistics")
    print("══════════════════════════════════════════════════════")
    print(f"  {'':20s}  {'Hybrid':>10}  {'Rolling LUT':>12}")
    for label, key in [("Bias (%)", "bias"), ("MAE (%)", "mae"),
                       ("RMSE (%)", "rmse"), ("P90 abs (%)", "p90"), ("Max abs (%)", "max")]:
        print(f"  {label:20s}  {ov['hybrid'].get(key,'—'):>10}  {ov['rolling'].get(key,'—'):>12}")
    if "mae_reduction_pct" in ov:
        print(f"\n  MAE reduction (hybrid vs rolling): {ov['mae_reduction_pct']:+.1f} %")
    m = s["modes"]
    print(f"\n  UNINIT={m.get('UNINIT',0)}  LUT={m.get('LUT',0)}  CC={m.get('CC',0)}"
          f"  resets/day={m.get('resets_per_day','—')}")
    rq = s.get("reset_quality", {})
    if rq:
        print(f"\n  Post-reset error: bias={rq.get('bias','—')} %  "
              f"MAE={rq.get('mae','—')} %  max={rq.get('max','—')} %")
    if s["by_day"]:
        print(f"\n  {'Day':>4}  {'N':>5}  {'Resets':>6}  "
              f"{'H-bias':>7}  {'H-MAE':>6}  {'R-bias':>7}  {'R-MAE':>6}")
        for r in s["by_day"]:
            print(f"  {r['day']:>4}  {r['n']:>5}  {r['n_resets']:>6}  "
                  f"{r['hybrid_bias']:>7.2f}  {r['hybrid_mae']:>6.2f}  "
                  f"{r['rolling_bias']:>7.2f}  {r['rolling_mae']:>6.2f}")
    print("══════════════════════════════════════════════════════\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
def build_comparison_plot(df: pl.DataFrame, serial: str, capacity: float) -> go.Figure:
    """Animated day-by-day: reported vs rolling vs hybrid + current."""
    days = sorted(df["day"].unique().to_list())

    def traces(day_df):
        x = day_df["sequence_number"].to_list()
        return [
            go.Scatter(x=x, y=day_df["mean_soc_rolling"].to_list(),
                       name="rolling LUT", mode="lines",
                       line=dict(color="#1F77B4", width=2.2)),
            go.Scatter(x=x, y=day_df["mean_soc_hybrid"].to_list(),
                       name="hybrid", mode="lines",
                       line=dict(color="#D62728", width=2.2)),
            go.Scatter(x=x, y=day_df["soc_nominal"].cast(pl.Float64).to_list(),
                       name="reported", mode="lines",
                       line=dict(color="black", width=2.8)),
            go.Scatter(x=x, y=day_df["current"].to_list(),
                       name="Current (A)", yaxis="y2", mode="lines",
                       line=dict(color="gray", width=1, dash="dot"), opacity=0.5),
        ]

    fig = go.Figure(data=traces(df.filter(pl.col("day") == days[0])))
    fig.frames = [
        go.Frame(name=str(d), data=traces(df.filter(pl.col("day") == d)))
        for d in days
    ]
    fig.update_layout(
        title=f"SoC Comparison — {serial} ({capacity} Ah)",
        yaxis=dict(title="SoC (%)", range=[-5, 105]),
        yaxis2=dict(title="Current (A)", overlaying="y", side="right", showgrid=False),
        xaxis_title="Sequence (within day)",
        hovermode="x unified", height=550,
        margin=dict(l=60, r=180, t=60, b=100),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.12, x=0.0,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, y=-0.12, x=0.05, len=0.9,
            currentvalue=dict(prefix="Day: "),
            steps=[dict(method="animate", label=str(d),
                        args=[[str(d)], dict(mode="immediate",
                               frame=dict(duration=0, redraw=True))])
                   for d in days],
        )],
    )
    return fig


def build_stats_plot(df: pl.DataFrame, s: dict, serial: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Error over time (estimated − reported)",
            "Error distribution",
            "MAE by SoC level",
            "CC drift: mean abs error vs minutes since last LUT reset",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )
    valid = (
        df.filter(pl.col("mean_soc_hybrid").is_not_null() & pl.col("soc_mode").ne("UNINIT"))
        .with_columns(soc_nom=pl.col("soc_nominal").cast(pl.Float64))
        .with_columns(
            err_hybrid =(pl.col("mean_soc_hybrid")  - pl.col("soc_nom")),
            err_rolling=(pl.col("mean_soc_rolling") - pl.col("soc_nom")),
        )
    )
    x_col = "sequence_number_original" if "sequence_number_original" in valid.columns else "sequence_number"
    x = valid[x_col].to_list()

    # Error over time + reset markers
    fig.add_trace(go.Scattergl(x=x, y=valid["err_hybrid"].to_list(),
                               name="Hybrid", mode="lines",
                               line=dict(color="#D62728", width=1), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scattergl(x=x, y=valid["err_rolling"].to_list(),
                               name="Rolling", mode="lines",
                               line=dict(color="#1F77B4", width=1), opacity=0.6), row=1, col=1)
    resets = df.filter(pl.col("soc_mode") == "LUT")
    if resets.height > 0:
        fig.add_trace(go.Scatter(
            x=resets[x_col].to_list(), y=[0] * resets.height,
            name="LUT reset", mode="markers",
            marker=dict(symbol="triangle-up", size=6, color="green", opacity=0.7),
        ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1), row=1, col=1)

    # Histograms
    fig.add_trace(go.Histogram(x=valid["err_hybrid"].drop_nulls().to_list(),
                               name="Hybrid", nbinsx=60, marker_color="#D62728",
                               opacity=0.6), row=1, col=2)
    fig.add_trace(go.Histogram(x=valid["err_rolling"].drop_nulls().to_list(),
                               name="Rolling", nbinsx=60, marker_color="#1F77B4",
                               opacity=0.5), row=1, col=2)

    # MAE by SoC
    if s["by_soc"]:
        bkts = [f"{int(r['soc_bucket'])}-{int(r['soc_bucket'])+10}%" for r in s["by_soc"]]
        fig.add_trace(go.Bar(x=bkts, y=[r["hybrid_mae"]  for r in s["by_soc"]],
                             name="Hybrid MAE",  marker_color="#D62728", opacity=0.8), row=2, col=1)
        fig.add_trace(go.Bar(x=bkts, y=[r["rolling_mae"] for r in s["by_soc"]],
                             name="Rolling MAE", marker_color="#1F77B4", opacity=0.6), row=2, col=1)

    # CC drift
    if s["cc_drift"]:
        mins = [r["steps"]        for r in s["cc_drift"]]
        errs = [r["mean_abs_err"] for r in s["cc_drift"]]
        fig.add_trace(go.Scatter(x=mins, y=errs, name="CC drift",
                                 mode="lines+markers",
                                 line=dict(color="#D62728", width=2)), row=2, col=2)

    fig.update_yaxes(title_text="Error (%)",        row=1, col=1)
    fig.update_yaxes(title_text="Count",            row=1, col=2)
    fig.update_yaxes(title_text="MAE (%)",          row=2, col=1)
    fig.update_yaxes(title_text="Mean abs err (%)", row=2, col=2)
    fig.update_xaxes(title_text="Sequence",         row=1, col=1)
    fig.update_xaxes(title_text="Error (%)",        row=1, col=2)
    fig.update_xaxes(title_text="SoC range",        row=2, col=1)
    fig.update_xaxes(title_text="Minutes since reset", row=2, col=2)
    fig.update_layout(
        title=f"Tracking Statistics — {serial}",
        barmode="group", height=800,
        margin=dict(l=70, r=60, t=80, b=60),
        legend=dict(x=1.01), hovermode="x unified",
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial",           required=True)
    p.add_argument("--capacity",         required=True, type=float)
    p.add_argument("--num_cells",        required=True, type=int)
    p.add_argument("--lut_path",         required=True)
    p.add_argument("--lut_soh",          default=100.0, type=float)
    p.add_argument("--aws_profile",      default="infra_dev")
    p.add_argument("--start_date",       default="2026-05-01")
    p.add_argument("--end_date",         default=today)
    p.add_argument("--tmp_dir",          default=None,
                   help="Directory for per-cell temp parquets (default: system tmp)")
    p.add_argument("--output_html",      default=None)
    p.add_argument("--stats_html",       default=None)
    p.add_argument("--output_csv",       default=None)
    args = p.parse_args()

    LOG.info("Loading LUT …")
    df_lut = load_lut(args.lut_path)

    df_raw = fetch_data(
        serial=args.serial, num_cells=args.num_cells,
        capacity_ah=args.capacity,
        start_date=args.start_date, end_date=args.end_date,
        aws_profile=args.aws_profile,
    )
    LOG.info("Fetched %d rows", len(df_raw))

    df_base = prepare_base(df_raw, args.capacity, args.num_cells)

    LOG.info("Running per-cell estimation (one cell at a time) …")
    df_wide = build_wide_result(
        df_base, df_lut,
        num_cells=args.num_cells,
        lut_soh=args.lut_soh,
        capacity_ah=args.capacity,
        tmp_dir=args.tmp_dir,
    )
    LOG.info("Wide result: %d rows", df_wide.height)

    if args.output_csv:
        df_wide.write_csv(args.output_csv)
        LOG.info("CSV → %s", args.output_csv)

    stats = compute_tracking_stats(df_wide)
    print_tracking_stats(stats)

    comp_fig  = build_comparison_plot(df_wide, args.serial, args.capacity)
    stats_fig = build_stats_plot(df_wide, stats, args.serial)

    if args.output_html:
        comp_fig.write_html(args.output_html, include_plotlyjs="cdn")
        LOG.info("Comparison HTML → %s", args.output_html)
    else:
        comp_fig.show()

    sh = args.stats_html or (
        args.output_html.replace(".html", "_stats.html") if args.output_html else None
    )
    if sh:
        stats_fig.write_html(sh, include_plotlyjs="cdn")
        LOG.info("Stats HTML → %s", sh)
    else:
        stats_fig.show()


if __name__ == "__main__":
    main()
