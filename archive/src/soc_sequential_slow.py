
"""
soc_sequential_slow.py
------------──────
A deliberately SLOW, SEQUENTIAL, ROW-BY-ROW SoC estimator.

Why this exists
------───────
The vectorised script (soc_hybrid_vs_reported.py) is fast but hides the
per-tick logic behind numpy tricks. This version mirrors what a live
embedded implementation will look like:

    For each timestamp (outer loop):
        For each cell (inner loop):
            1. Bin temperature
            2. Bin current
            3. Look up (temp_bin, current_bin) in the LUT
                  if present  -> LUT reset, compute soc_min/max/mid from V↔DoD
                  if absent   -> CC mode, integrate delta_charge into prev SoC
                                 (or UNINIT until the first LUT hit)
            4. Log the entire decision trail for this (row, cell) pair.

Nothing is vectorised. Nothing is clever. Every line is meant to be
readable by someone who just started writing software.

Inputs
──────
CSV columns (raw):
    sequence_number, time_stamp,
    temperature_battery, current,
    cell_voltage_1 ... cell_voltage_N,
    charge_in, charge_out, soc

LUT parquet: same as the vectorised script. Loaded via the existing
load_lut() pipeline so the SAME (temp, current) validity rules apply.

Usage
─────
uv run soc_sequential_slow.py \\
    --csv         data/raw.csv \\
    --lut_path    data/lut.parquet \\
    --capacity    277 \\
    --num_cells   32 \\
    --resistance_ohm 0.00025 \\
    --output_csv  soc_sequential_output.csv \\
    --log_file    soc_sequential.log
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

import polars as pl


# ── Constants (kept simple and visible) ------------------------───────
LUT_REFERENCE_CAPACITY = 277.0
TEMP_BIN_SIZE = 5.0      # bin temperature_battery into multiples of 5
CURRENT_BIN_SIZE = 5.0   # bin scaled current into multiples of 5
SECONDS_PER_TICK = 60.0  # delta_charge is per-60s window (matches Athena cadence)


# ── Logging setup (default for any script per user preference) ------------
def setup_logging(log_file: str) -> logging.Logger:
    """One shared logger. File gets EVERYTHING, stderr gets a summary."""
    logger = logging.getLogger("soc_seq")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler — verbose, every per-row block goes here
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Stderr handler — quieter, just INFO+ so the terminal isn't a wall of text
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


# ── LUT loader (same filtering as the vectorised script) ------------──────
def load_lut(lut_path: str, log: logging.Logger) -> pl.DataFrame:
    """
    Load the parquet LUT and run the same filtering pipeline as
    soc_hybrid_vs_reported.py so the validity rules match.

    The returned DataFrame has columns:
        cell_temperatures, current_average, dod, cell_voltages_mean
    Every (cell_temperatures, current_average) pair present here is a
    valid reset point.
    """
    log.info("Loading LUT from %s", lut_path)
    df = (
        pl.scan_parquet(lut_path)
        .with_columns((pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod"))
        .filter(pl.col("dod").is_between(0, 100, closed="both"))
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
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
        .filter(pl.col("soh").is_between(97, 103, closed="both"))
        .group_by(["cell_temperatures", "current_average", "dod"])
        .agg((pl.col("cell_voltages_mean").mean()).round(0).alias("cell_voltages_mean"))
        .with_columns(
            pl.col("dod")
            .count()
            .over(["cell_temperatures", "current_average"])
            .alias("dod_count")
        )
        .filter(pl.col("dod_count") >= 90)
        .with_columns(
            pl.col("cell_voltages_mean")
            .diff()
            .over(
                partition_by=["current_average", "cell_temperatures"],
                order_by="dod",
                descending=True,
            )
            .alias("cell_voltages_diff"),
            pl.col("dod")
            .diff()
            .over(
                partition_by=["current_average", "cell_temperatures"],
                order_by="dod",
                descending=False,
            )
            .alias("dod_diff"),
        )
        .filter((pl.col("dod_diff").abs() == 1) | (pl.col("dod") == 0))
        .filter((pl.col("cell_voltages_diff").abs() >= 5) | (pl.col("dod") == 0))
        .filter(pl.col("current_average")>20)
        .collect()
    )
    log.info("LUT loaded: %d rows", df.height)
    return df


# ── Convert LUT into a plain Python dict for sequential lookup ------------
def build_lut_lookup(
    df_lut: pl.DataFrame, log: logging.Logger
) -> dict[tuple[float, float], list[tuple[float, float]]]:
    """
    Convert the LUT DataFrame into a dictionary:

        key   = (temp_bin, current_bin)
        value = sorted list of (dod, voltage_mv) tuples, ascending by dod

    Sequential code can then do a plain `if key in lut: ...` check.
    Voltage in the LUT is in millivolts (matches *1000 conversion in raw script).
    """
    lookup: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in df_lut.iter_rows(named=True):
        key = (float(row["cell_temperatures"]), float(row["current_average"]))
        lookup.setdefault(key, []).append(
            (float(row["dod"]), float(row["cell_voltages_mean"]))
        )
    # Keep each curve sorted by dod so interpolation is straightforward
    for key in lookup:
        lookup[key].sort(key=lambda x: x[0])
    log.info("LUT lookup table built: %d valid (temp, current) keys", len(lookup))
    return lookup


# ── Tiny helpers — one job each ------------------------------───────
def bin_value(value: float, bin_size: float) -> float:
    """Floor-bin a value to a multiple of bin_size. e.g. 23.4, 5 -> 20.0"""
    return (value // bin_size) * bin_size


def scale_current(raw_current: float, capacity_ah: float) -> float:
    """Scale current to the LUT's reference capacity (277 Ah)."""
    return raw_current / capacity_ah * LUT_REFERENCE_CAPACITY


def match_soc_within_tolerance(
    curve: list[tuple[float, float]],
    v_adjusted_mv: float,
    voltage_tolerance_mv: float,
) -> tuple[float, float, float] | None:
    """
    Look for the LUT point in this V↔DoD curve whose voltage is closest to
    v_adjusted_mv, but ONLY if |v_lut - v_adjusted| <= voltage_tolerance_mv.

    No interpolation. No bracketing. Just a direct point match.

    Returns (soc_min, soc_max, soc_mid) where all three are equal to
    100 - dod_of_closest_point, OR None if nothing is within tolerance.
    """
    best_dod: float | None = None
    best_abs_diff: float = float("inf")
    for dod, volt in curve:
        diff = abs(volt - v_adjusted_mv)
        if diff <= voltage_tolerance_mv and diff < best_abs_diff:
            best_abs_diff = diff
            best_dod = dod
    if best_dod is None:
        return None
    soc = 100.0 - best_dod
    soc = max(0.0, min(100.0, soc))
    return soc, soc, soc


# ── The per-tick state for one cell (plain dataclass-ish dict) ------------
def make_cell_state() -> dict:
    """A live cell tracker is just: 'what was my SoC last tick?' + 'am I initialised?'"""
    return {
        "prev_soc": None,         # last good SoC estimate (None until first LUT hit)
        "initialized": False,     # True once we've ever had a LUT reset
        "reset_count": 0,         # how many LUT resets seen so far (handy for stats)
        "prev_voltage_mv": None,  # cell voltage at the previous tick (mV), None on first
    }


# ── The whole per-(row, cell) decision, written as simply as possible ------─
def _run_cc_or_uninit(
    log: logging.Logger,
    seq: int,
    cell_idx: int,
    delta_charge: float | None,
    state: dict,
    capacity_ah: float,
    reason: str,
) -> dict:
    """
    Run the CC update for this tick — or return UNINIT if we've never
    seen a LUT reset yet. Shared by:
       - rows whose (temp, current) is not in the LUT
       - rows whose voltage is not within tolerance of any LUT point
    `reason` is a short string written to the log so you can tell the
    two cases apart.
    """
    if not state["initialized"]:
        log.debug("    state.initialized=False → mode=UNINIT  (reason: %s)", reason)
        log.debug("    soc_hybrid = None  (no estimate yet)")
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "UNINIT", "soc_min": None, "soc_max": None,
            "soc_mid": None, "soc_hybrid": None,
        }
    dc = 0.0 if delta_charge is None else float(delta_charge)
    d_soc = dc / (capacity_ah * 3600.0) * 100.0
    new_soc = state["prev_soc"] + d_soc
    new_soc_clamped = max(0.0, min(100.0, new_soc))
    log.debug("    state.initialized=True → mode=CC  (reason: %s)", reason)
    log.debug("    prev_soc       = %.4f %%", state["prev_soc"])
    log.debug("    delta_charge   = %.4f C", dc)
    log.debug("    d_soc          = delta_charge / (capacity_Ah * 3600) * 100")
    log.debug("                   = %.4f / (%.1f * 3600) * 100 = %.6f %%",
              dc, capacity_ah, d_soc)
    log.debug("    new_soc        = prev_soc + d_soc = %.4f %%", new_soc)
    if new_soc != new_soc_clamped:
        log.debug("    CLAMPED to [0, 100] → %.4f %%", new_soc_clamped)
    state["prev_soc"] = new_soc_clamped
    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": "CC", "soc_min": None, "soc_max": None,
        "soc_mid": None, "soc_hybrid": new_soc_clamped,
    }


def process_one_cell_tick(
    log: logging.Logger,
    seq: int,
    time_stamp,
    cell_idx: int,
    cell_voltage_v: float | None,
    raw_current: float | None,
    raw_temp: float | None,
    delta_charge: float | None,
    state: dict,
    lut_lookup: dict,
    capacity_ah: float,
    resistance_ohm: float,
    ir_compensation: bool,
    voltage_tolerance_mv: float,
    voltage_delta_min_mv: float,
) -> dict:
    """
    Run ONE cell's update for ONE timestamp. Returns a dict with the
    decision trail so the caller can write it to the output CSV.

    Logs a multi-line block to the log file describing every step.
    """
    log.debug("─" * 70)
    log.debug("ROW seq=%d  time=%s  cell=%d", seq, time_stamp, cell_idx)
    log.debug("─" * 70)

    # ── Step 1: read raw inputs ------------------------------──
    log.debug("[1] RAW INPUTS")
    log.debug("    cell_voltage_V = %s", cell_voltage_v)
    log.debug("    current_A      = %s", raw_current)
    log.debug("    temperature_C  = %s", raw_temp)
    log.debug("    delta_charge_C = %s   (coulombs in this 60s tick)", delta_charge)

    # ── Step 2: validate ------------------------------------─
    log.debug("[2] VALIDATE")
    if cell_voltage_v is None or cell_voltage_v <= 0:
        log.debug("    cell_voltage missing/<=0 → skip this cell for this tick")
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "SKIP", "soc_min": None, "soc_max": None,
            "soc_mid": None, "soc_hybrid": state["prev_soc"],
        }
    if raw_current is None or raw_temp is None:
        log.debug("    current or temperature missing → skip")
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "SKIP", "soc_min": None, "soc_max": None,
            "soc_mid": None, "soc_hybrid": state["prev_soc"],
        }
    log.debug("    inputs OK")

    # ── Step 3: bin temperature and current ------------------──────
    log.debug("[3] BIN")
    temp_bin = bin_value(float(raw_temp), TEMP_BIN_SIZE)
    current_scaled = scale_current(float(raw_current), capacity_ah)
    current_bin = bin_value(current_scaled, CURRENT_BIN_SIZE)
    log.debug("    temp_bin     = %.1f  (from %.2f, bin size %.0f)",
              temp_bin, raw_temp, TEMP_BIN_SIZE)
    log.debug("    current_scaled = %.2f  (raw %.2f * %.0f / %.0f)",
              current_scaled, raw_current, LUT_REFERENCE_CAPACITY, capacity_ah)
    log.debug("    current_bin  = %.1f  (bin size %.0f)",
              current_bin, CURRENT_BIN_SIZE)

    # ── Step 4: look up in LUT ------------------------------───
    log.debug("[4] LUT LOOKUP")
    key = (temp_bin, current_bin)
    cell_voltage_mv = cell_voltage_v * 1000.0
    log.debug("    key = (temp_bin=%.1f, current_bin=%.1f)", *key)
    log.debug("    cell_voltage_mV = %.2f", cell_voltage_mv)

    # ── Step 4.5: voltage-delta gate ------------------------─────
    # Require the cell voltage to have moved by at least voltage_delta_min_mv
    # since the previous tick before we even attempt a LUT reset. This
    # filters out flat regions where consecutive voltages look identical
    # and an LUT hit would just be noise. Failing this gate falls through
    # to CC just like a missing key would.
    log.debug("[4.5] VOLTAGE DELTA GATE (require |v[N] - v[N-1]| >= %.2f mV)",
              voltage_delta_min_mv)
    prev_v = state["prev_voltage_mv"]
    if prev_v is None:
        log.debug("    no previous voltage (first tick for this cell) → fall through to CC")
        state["prev_voltage_mv"] = cell_voltage_mv
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason="first tick — no prev voltage",
        )
    v_delta = cell_voltage_mv - prev_v
    log.debug("    prev_voltage_mV = %.2f", prev_v)
    log.debug("    v_delta_mV      = cell_voltage_mV - prev_voltage_mV = %.2f", v_delta)
    if abs(v_delta) < voltage_delta_min_mv:
        log.debug("    |v_delta| < threshold → no reset, fall through to CC")
        state["prev_voltage_mv"] = cell_voltage_mv
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason="voltage delta below threshold",
        )
    log.debug("    |v_delta| >= threshold → proceed to LUT lookup")
    # Stash for next tick — every path past here also returns, so set it now
    # so we don't accidentally forget it on the LUT-reset return.
    state["prev_voltage_mv"] = cell_voltage_mv

    if key not in lut_lookup:
        # ── Step 5a: no LUT slice for this (temp, current) ------───────
        log.debug("[5] DECISION: key NOT in LUT → no reset this tick")
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason="key not in LUT",
        )

    # ── Step 5b: LUT reset ------------------------------───────
    log.debug("[5] DECISION: key IN LUT → LUT reset this tick")
    curve = lut_lookup[key]
    log.debug("    LUT curve has %d points for this (temp, current)", len(curve))

    # IR compensation: at this row's current we measured cell_voltage,
    # but the LUT curve was recorded at a fixed current_bin. Add back the
    # IR drop difference so we compare like-for-like.
    # Toggleable — when off, we use the raw cell voltage directly.
    log.debug("[6] IR COMPENSATION")
    if ir_compensation:
        r_mohm = resistance_ohm * 1000.0  # ohms → milli-ohms (V is in mV)
        delta_current = current_scaled - current_bin
        v_adjusted_mv = cell_voltage_mv - r_mohm * delta_current
        log.debug("    enabled")
        log.debug("    r_milliohm     = %.3f mΩ", r_mohm)
        log.debug("    delta_current  = current_scaled - current_bin")
        log.debug("                   = %.2f - %.1f = %.2f A",
                  current_scaled, current_bin, delta_current)
        log.debug("    v_adjusted_mV  = cell_voltage_mV - r_mΩ * delta_current")
        log.debug("                   = %.2f - %.3f * %.2f = %.2f mV",
                  cell_voltage_mv, r_mohm, delta_current, v_adjusted_mv)
    else:
        v_adjusted_mv = cell_voltage_mv
        log.debug("    disabled → v_adjusted_mV = cell_voltage_mV = %.2f mV",
                  v_adjusted_mv)

    match = match_soc_within_tolerance(curve, v_adjusted_mv, voltage_tolerance_mv)
    log.debug("[7] VOLTAGE MATCH (±%.2f mV, no interpolation)", voltage_tolerance_mv)
    if match is None:
        log.debug("    no LUT point within tolerance → no reset, fall through to CC")
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason="no LUT voltage within tolerance",
        )
    soc_min, soc_max, soc_mid = match
    log.debug("    matched! soc_min = soc_max = soc_mid = %.4f %%", soc_mid)

    # ── Update state ------------------------------------────
    state["prev_soc"] = soc_mid
    state["initialized"] = True
    state["reset_count"] += 1
    log.debug("[8] STATE AFTER")
    log.debug("    prev_soc       = %.4f %%", state["prev_soc"])
    log.debug("    initialized    = True")
    log.debug("    reset_count    = %d", state["reset_count"])

    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": "LUT", "soc_min": soc_min, "soc_max": soc_max,
        "soc_mid": soc_mid, "soc_hybrid": soc_mid,
    }


# ── Outer driver: row by row, cell by cell ------------------------────
def run_sequential(
    log: logging.Logger,
    csv_path: str,
    lut_lookup: dict,
    num_cells: int,
    capacity_ah: float,
    resistance_ohm: float,
    ir_compensation: bool,
    voltage_tolerance_mv: float,
    voltage_delta_min_mv: float,
    output_csv: str,
) -> None:
    """
    Outer loop: every timestamp.
    Inner loop: every cell.

    This is the embedded-style tick model — at each new tick we update
    every cell, then move on. Nothing is vectorised.
    """
    log.info("Reading CSV: %s", csv_path)
    df = pl.read_csv(csv_path, try_parse_dates=True).sort("sequence_number")
    log.info("CSV rows: %d", df.height)

    # Sanity check the columns
    needed = {
        "sequence_number", "time_stamp", "temperature_battery",
        "current", "charge_in", "charge_out", "soc",
    }
    cell_cols = [f"cell_voltage_{i}" for i in range(1, num_cells + 1)]
    needed.update(cell_cols)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Build delta_charge per tick (charge_in_diff - charge_out_diff).
    # This matches prepare_base() in the vectorised script.
    df = df.with_columns(
        delta_charge=(
            (pl.col("charge_in").diff() - pl.col("charge_out").diff())
            .cast(pl.Float64)
        )
    )

    # Initialise per-cell state once
    cell_states = [make_cell_state() for _ in range(num_cells)]
    log.info("Initialised state for %d cells", num_cells)

    # Per-row output rows
    out_rows: list[dict] = []

    rows = df.to_dicts()
    total = len(rows)
    log_progress_every = max(1, total // 20)

    for row_idx, row in enumerate(rows):
        if row_idx % log_progress_every == 0:
            log.info("Processing row %d / %d  (seq=%s)",
                     row_idx, total, row["sequence_number"])

        log.debug("=" * 70)
        log.debug("TICK %d / %d   seq=%s   time=%s",
                  row_idx, total, row["sequence_number"], row["time_stamp"])
        log.debug("    pack temperature=%s  pack current=%s  reported_soc=%s",
                  row["temperature_battery"], row["current"], row["soc"])
        log.debug("=" * 70)

        # Inner loop: every cell, this tick
        for i in range(num_cells):
            cell_idx = i + 1
            result = process_one_cell_tick(
                log=log,
                seq=row["sequence_number"],
                time_stamp=row["time_stamp"],
                cell_idx=cell_idx,
                cell_voltage_v=row[f"cell_voltage_{cell_idx}"],
                raw_current=row["current"],
                raw_temp=row["temperature_battery"],
                delta_charge=row["delta_charge"],
                state=cell_states[i],
                lut_lookup=lut_lookup,
                capacity_ah=capacity_ah,
                resistance_ohm=resistance_ohm,
                ir_compensation=ir_compensation,
                voltage_tolerance_mv=voltage_tolerance_mv,
                voltage_delta_min_mv=voltage_delta_min_mv,
            )
            result["time_stamp"] = str(row["time_stamp"])
            result["soc_nominal"] = row["soc"]
            out_rows.append(result)

    log.info("Done. Writing %d (row, cell) entries to %s", len(out_rows), output_csv)
    # Explicit schema: the first ~UNINIT rows have all-None numeric columns,
    # which causes polars to infer Null dtype and then fail on later float
    # values. Pin the dtypes here so the float columns stay Float64.
    out_schema = {
        "sequence_number": pl.Int64,
        "cell_idx":        pl.Int64,
        "mode":            pl.Utf8,
        "soc_min":         pl.Float64,
        "soc_max":         pl.Float64,
        "soc_mid":         pl.Float64,
        "soc_hybrid":      pl.Float64,
        "time_stamp":      pl.Utf8,    # written as-is; CSV anyway
        "soc_nominal":     pl.Float64,
    }
    pl.DataFrame(out_rows, schema=out_schema).write_csv(output_csv)

    # Per-cell summary
    log.info("Per-cell summary:")
    for i, st in enumerate(cell_states, start=1):
        log.info("  cell %2d: reset_count=%d  final_soc=%s",
                 i, st["reset_count"], st["prev_soc"])


# ── Entry point ------------------------------------------───────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",            required=True, help="Input CSV path")
    p.add_argument("--lut_path",       required=True, help="LUT parquet path")
    p.add_argument("--capacity",       required=True, type=float, help="Pack capacity Ah")
    p.add_argument("--num_cells",      required=True, type=int)
    p.add_argument("--resistance_ohm", default=0.00025, type=float)
    p.add_argument("--ir_compensation", action="store_true", default=False,
                   help="Apply IR compensation to cell voltage before LUT lookup "
                        "(default OFF). Pass this flag to enable.")
    p.add_argument("--voltage_tolerance_mv", default=2.0, type=float,
                   help="LUT reset only fires when |v_cell - v_lut| <= this many mV "
                        "(default 2). No interpolation; closest match within tolerance wins.")
    p.add_argument("--voltage_delta_min_mv", default=5.0, type=float,
                   help="LUT reset is gated on |v_cell[N] - v_cell[N-1]| >= this many mV "
                        "(default 5). Below threshold falls through to CC. First tick of "
                        "each cell has no previous voltage and also falls through.")
    p.add_argument("--output_csv",     default="soc_sequential_output.csv")
    p.add_argument("--log_file",       default="soc_sequential.log")
    args = p.parse_args()

    log = setup_logging(args.log_file)
    log.info("------ soc_sequential_slow start ------")
    log.info("Args: %s", vars(args))

    df_lut = load_lut(args.lut_path, log)
    lut_lookup = build_lut_lookup(df_lut, log)

    if not Path(args.csv).exists():
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)

    run_sequential(
        log=log,
        csv_path=args.csv,
        lut_lookup=lut_lookup,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        resistance_ohm=args.resistance_ohm,
        ir_compensation=args.ir_compensation,
        voltage_tolerance_mv=args.voltage_tolerance_mv,
        voltage_delta_min_mv=args.voltage_delta_min_mv,
        output_csv=args.output_csv,
    )
    log.info("------ soc_sequential_slow done ------")


if __name__ == "__main__":
    main()
