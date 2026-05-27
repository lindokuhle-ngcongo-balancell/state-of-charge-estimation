# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "numpy"]
# ///
"""
soc_sequential_ocv.py
─────────────────────
Sequential, row-by-row SoC estimator using an OCV (open-circuit voltage)
lookup table.

Key differences from soc_sequential_slow.py
────────────────────────────────────────────
* LUT is a simple CSV with columns  soc, voltage  (voltage in mV).
  No temperature or current binning — OCV has neither dependency here.
* LUT reset is gated on the pack current having stayed within ±REST_CURRENT_A
  for at least REST_MIN_TICKS consecutive ticks (default 5 × 60 s = 5 min).
  The gate is pack-level (one current reading per row, series pack).
* After a LUT reset fires the rest counter resets to zero; the cell must
  earn another full rest window before the next reset.
* SoC is found by linear interpolation between the two bracketing OCV points.
* Voltages above the 100 % OCV point → SoC = 100 %.
* Voltages below the   0 % OCV point → SoC =   0 %.
* Voltage-delta gate is optional (--voltage_delta_min_mv, default disabled).
* IR compensation is removed (not meaningful for OCV).

Inputs
──────
CSV columns (raw):
    sequence_number, time_stamp,
    temperature_battery, current,
    cell_voltage_1 … cell_voltage_N,
    charge_in, charge_out, soc

OCV LUT CSV columns:
    soc      — 0 … 100 (%)
    voltage  — mV

Usage
─────
uv run soc_sequential_ocv.py \\
    --csv              data/raw.csv \\
    --lut_csv          lut_soh_eq_100_ocv_soc.csv \\
    --capacity         277 \\
    --num_cells        32 \\
    --output_csv       soc_ocv_output.csv \\
    --log_file         soc_sequential_ocv.log

Optional flags:
    --rest_current_a        1.0     threshold for ±rest window (A)
    --rest_min_ticks        5       consecutive ticks required (×60 s each)
    --voltage_tolerance_mv  2.0     if set, LUT reset only fires when the
                                    interpolated lookup is within this many mV
                                    of an actual LUT point (0 = disabled)
    --voltage_delta_min_mv  0.0     LUT reset gated on |v[N]-v[N-1]| >= this
                                    (0.0 = disabled, which is the default)
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

import polars as pl


# ── Constants ────────────────────────────────────────────────────────────────
SECONDS_PER_TICK = 60.0   # Athena cadence


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> logging.Logger:
    """File gets everything; stderr gets INFO+."""
    logger = logging.getLogger("soc_ocv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


# ── OCV LUT loader ────────────────────────────────────────────────────────────
def load_ocv_lut(lut_csv: str, log: logging.Logger) -> list[tuple[float, float]]:
    """
    Load the OCV CSV and return a list of (voltage_mv, soc_pct) tuples
    sorted ascending by voltage.  We sort by voltage because interpolation
    will be voltage-in → SoC-out.

    The CSV has columns  soc (0–100 %)  and  voltage (mV).
    """
    log.info("Loading OCV LUT from %s", lut_csv)
    df = pl.read_csv(lut_csv)

    required = {"soc", "voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OCV LUT CSV missing columns: {sorted(missing)}")

    # Sort ascending by voltage so interpolation brackets work cleanly
    df = df.sort("voltage")
    curve: list[tuple[float, float]] = [
        (float(r["voltage"]), float(r["soc"]))
        for r in df.iter_rows(named=True)
    ]
    log.info(
        "OCV LUT loaded: %d points  voltage range [%.0f, %.0f] mV  SoC range [%.1f, %.1f] %%",
        len(curve),
        curve[0][0], curve[-1][0],
        curve[0][1], curve[-1][1],
    )
    return curve


# ── OCV interpolation ─────────────────────────────────────────────────────────
def ocv_voltage_to_soc(
    curve: list[tuple[float, float]],
    voltage_mv: float,
) -> float:
    """
    Linear interpolation:  voltage_mv  →  SoC (%).

    Curve is sorted ascending by voltage.
    * voltage above the highest LUT point  → 100.0 %
    * voltage below the lowest  LUT point  → 0.0 %
    * otherwise linear interpolation between the two bracketing points.
    """
    v_min, soc_at_vmin = curve[0]
    v_max, soc_at_vmax = curve[-1]

    if voltage_mv >= v_max:
        return 100.0
    if voltage_mv <= v_min:
        return 0.0

    # Binary-search for the lower bracket
    lo, hi = 0, len(curve) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if curve[mid][0] <= voltage_mv:
            lo = mid
        else:
            hi = mid

    v_lo, soc_lo = curve[lo]
    v_hi, soc_hi = curve[hi]

    # Guard against duplicate voltage entries (shouldn't happen in a clean LUT)
    if v_hi == v_lo:
        return float(soc_lo)

    fraction = (voltage_mv - v_lo) / (v_hi - v_lo)
    soc = soc_lo + fraction * (soc_hi - soc_lo)
    return max(0.0, min(100.0, soc))


# ── Per-cell state ────────────────────────────────────────────────────────────
def make_cell_state() -> dict:
    return {
        "prev_soc":        None,   # last SoC estimate; None until first LUT reset
        "initialized":     False,
        "reset_count":     0,
        "prev_voltage_mv": None,   # cell voltage last tick (mV)
    }


# ── CC fallback (or UNINIT) ───────────────────────────────────────────────────
def _run_cc_or_uninit(
    log: logging.Logger,
    seq: int,
    cell_idx: int,
    delta_charge: float | None,
    state: dict,
    capacity_ah: float,
    reason: str,
) -> dict:
    if not state["initialized"]:
        log.debug("    initialized=False → UNINIT  (reason: %s)", reason)
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "UNINIT",
            "soc_min": None, "soc_max": None, "soc_mid": None, "soc_hybrid": None,
        }
    dc = 0.0 if delta_charge is None else float(delta_charge)
    d_soc = dc / (capacity_ah * 3600.0) * 100.0
    new_soc = state["prev_soc"] + d_soc
    new_soc_clamped = max(0.0, min(100.0, new_soc))
    log.debug("    initialized=True → CC  (reason: %s)", reason)
    log.debug("    prev_soc=%.4f%%  delta_charge=%.4fC  d_soc=%.6f%%  new_soc=%.4f%%",
              state["prev_soc"], dc, d_soc, new_soc_clamped)
    state["prev_soc"] = new_soc_clamped
    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": "CC",
        "soc_min": None, "soc_max": None, "soc_mid": None, "soc_hybrid": new_soc_clamped,
    }


# ── Per-(tick, cell) decision ─────────────────────────────────────────────────
def process_one_cell_tick(
    log: logging.Logger,
    seq: int,
    time_stamp,
    cell_idx: int,
    cell_voltage_v: float | None,
    delta_charge: float | None,
    state: dict,
    ocv_curve: list[tuple[float, float]],
    capacity_ah: float,
    rest_qualified: bool,           # True when pack has rested ≥ REST_MIN_TICKS
    voltage_tolerance_mv: float,    # 0.0 = disabled
    voltage_delta_min_mv: float,    # 0.0 = disabled
) -> dict:
    """
    One cell, one tick.  Returns a result dict for the output CSV.

    rest_qualified is a pack-level boolean computed by the outer loop
    before this function is called — no current data needed here.
    """
    log.debug("─" * 70)
    log.debug("ROW seq=%d  time=%s  cell=%d", seq, time_stamp, cell_idx)
    log.debug("─" * 70)

    # ── Step 1: validate ─────────────────────────────────────────────
    log.debug("[1] VALIDATE")
    if cell_voltage_v is None or cell_voltage_v <= 0:
        log.debug("    cell_voltage missing/invalid → SKIP")
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "SKIP",
            "soc_min": None, "soc_max": None, "soc_mid": None,
            "soc_hybrid": state["prev_soc"],
        }
    log.debug("    inputs OK")

    cell_voltage_mv = cell_voltage_v * 1000.0
    log.debug("    cell_voltage_mV = %.2f", cell_voltage_mv)

    # ── Step 2: rest window gate ──────────────────────────────────────
    log.debug("[2] REST WINDOW GATE")
    if not rest_qualified:
        log.debug("    pack not at rest long enough → CC/UNINIT")
        state["prev_voltage_mv"] = cell_voltage_mv   # still track voltage
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason="pack not rested",
        )
    log.debug("    rest window qualified → proceed")

    # ── Step 3: optional voltage-delta gate ──────────────────────────
    log.debug("[3] VOLTAGE DELTA GATE (threshold=%.2f mV)", voltage_delta_min_mv)
    prev_v = state["prev_voltage_mv"]
    state["prev_voltage_mv"] = cell_voltage_mv   # update for next tick regardless

    if voltage_delta_min_mv > 0.0:
        if prev_v is None:
            log.debug("    no previous voltage (first tick) → CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="first tick — no prev voltage",
            )
        v_delta = abs(cell_voltage_mv - prev_v)
        log.debug("    prev_voltage_mV=%.2f  v_delta=%.2f mV", prev_v, v_delta)
        if v_delta < voltage_delta_min_mv:
            log.debug("    |v_delta| < threshold → CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="voltage delta below threshold",
            )
        log.debug("    |v_delta| >= threshold → proceed")
    else:
        log.debug("    voltage delta gate disabled → proceed")

    # ── Step 4: OCV interpolation ─────────────────────────────────────
    log.debug("[4] OCV INTERPOLATION")
    soc_interp = ocv_voltage_to_soc(ocv_curve, cell_voltage_mv)
    log.debug("    interpolated SoC = %.4f %%", soc_interp)

    # ── Step 5: optional voltage-tolerance gate ───────────────────────
    # When enabled: find the nearest LUT voltage; if the cell voltage is
    # further than voltage_tolerance_mv from it, refuse the LUT reset.
    log.debug("[5] VOLTAGE TOLERANCE GATE (threshold=%.2f mV)", voltage_tolerance_mv)
    if voltage_tolerance_mv > 0.0:
        nearest_dist = min(abs(cell_voltage_mv - v) for v, _ in ocv_curve)
        log.debug("    nearest LUT voltage distance = %.2f mV", nearest_dist)
        if nearest_dist > voltage_tolerance_mv:
            log.debug("    distance > tolerance → reject LUT reset, CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="cell voltage outside tolerance of nearest OCV point",
            )
        log.debug("    within tolerance → accept")
    else:
        log.debug("    tolerance gate disabled → accept")

    # ── Step 6: LUT reset ─────────────────────────────────────────────
    log.debug("[6] LUT RESET")
    soc_interp = max(0.0, min(100.0, soc_interp))
    state["prev_soc"] = soc_interp
    state["initialized"] = True
    state["reset_count"] += 1
    log.debug("    soc = %.4f %%  reset_count = %d", soc_interp, state["reset_count"])

    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": "LUT",
        "soc_min": soc_interp, "soc_max": soc_interp, "soc_mid": soc_interp,
        "soc_hybrid": soc_interp,
    }


# ── Pack-level rest tracker ───────────────────────────────────────────────────
def make_rest_state(rest_current_a: float, rest_min_ticks: int) -> dict:
    """
    Tracks how many consecutive ticks the pack current has been within
    ±rest_current_a.  When consecutive_rest_ticks >= rest_min_ticks the
    pack is 'qualified' for an OCV reset.

    After a reset fires the caller must call reset_after_lut_fire() to
    zero the counter so a fresh rest window is required.
    """
    return {
        "rest_current_a":    rest_current_a,
        "rest_min_ticks":    rest_min_ticks,
        "consecutive_ticks": 0,
    }


def update_rest_state(state: dict, current_a: float | None) -> bool:
    """
    Call once per tick with the pack current.
    Returns True when the rest window is qualified this tick.
    current_a=None is treated as out-of-rest (unknown).
    """
    if current_a is None or abs(float(current_a)) > state["rest_current_a"]:
        state["consecutive_ticks"] = 0
        return False
    state["consecutive_ticks"] += 1
    return state["consecutive_ticks"] >= state["rest_min_ticks"]


def reset_rest_after_lut_fire(state: dict) -> None:
    """Call after a LUT reset has fired so the next reset needs a fresh window."""
    state["consecutive_ticks"] = 0


# ── Outer driver ──────────────────────────────────────────────────────────────
def run_sequential(
    log: logging.Logger,
    csv_path: str,
    ocv_curve: list[tuple[float, float]],
    num_cells: int,
    capacity_ah: float,
    rest_current_a: float,
    rest_min_ticks: int,
    voltage_tolerance_mv: float,
    voltage_delta_min_mv: float,
    output_csv: str,
) -> None:
    log.info("Reading CSV: %s", csv_path)
    df = pl.read_csv(csv_path, try_parse_dates=True).sort("sequence_number")
    log.info("CSV rows: %d", df.height)

    needed = {
        "sequence_number", "time_stamp", "temperature_battery",
        "current", "charge_in", "charge_out", "soc",
    }
    cell_cols = [f"cell_voltage_{i}" for i in range(1, num_cells + 1)]
    needed.update(cell_cols)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.with_columns(
        delta_charge=(
            (pl.col("charge_in").diff() - pl.col("charge_out").diff())
            .cast(pl.Float64)
        )
    )

    cell_states = [make_cell_state() for _ in range(num_cells)]
    rest_state = make_rest_state(rest_current_a, rest_min_ticks)
    log.info("Initialised state for %d cells", num_cells)
    log.info(
        "Rest window: ±%.2f A for %d consecutive ticks (%.0f min)",
        rest_current_a, rest_min_ticks, rest_min_ticks * SECONDS_PER_TICK / 60.0,
    )

    out_rows: list[dict] = []
    rows = df.to_dicts()
    total = len(rows)
    log_progress_every = max(1, total // 20)

    for row_idx, row in enumerate(rows):
        if row_idx % log_progress_every == 0:
            log.info("Processing row %d / %d  (seq=%s)",
                     row_idx, total, row["sequence_number"])

        seq = row["sequence_number"]
        ts = row["time_stamp"]
        raw_current = row["current"]

        # ── Pack-level rest window update (one per tick, before cell loop) ──
        rest_qualified = update_rest_state(rest_state, raw_current)

        log.debug("=" * 70)
        log.debug("TICK %d / %d  seq=%s  time=%s", row_idx, total, seq, ts)
        log.debug("    current=%.3f A  rest_ticks=%d/%d  qualified=%s",
                  float(raw_current) if raw_current is not None else float("nan"),
                  rest_state["consecutive_ticks"], rest_state["rest_min_ticks"],
                  rest_qualified)
        log.debug("=" * 70)

        # Track whether ANY cell fires a LUT reset this tick so we can
        # reset the rest counter exactly once after the cell loop.
        any_lut_reset_this_tick = False

        for i in range(num_cells):
            cell_idx = i + 1
            result = process_one_cell_tick(
                log=log,
                seq=seq,
                time_stamp=ts,
                cell_idx=cell_idx,
                cell_voltage_v=row[f"cell_voltage_{cell_idx}"],
                delta_charge=row["delta_charge"],
                state=cell_states[i],
                ocv_curve=ocv_curve,
                capacity_ah=capacity_ah,
                rest_qualified=rest_qualified,
                voltage_tolerance_mv=voltage_tolerance_mv,
                voltage_delta_min_mv=voltage_delta_min_mv,
            )
            if result["mode"] == "LUT":
                any_lut_reset_this_tick = True

            result["time_stamp"] = str(ts)
            result["soc_nominal"] = row["soc"]
            out_rows.append(result)

        # Reset rest counter after any LUT fire this tick
        if any_lut_reset_this_tick:
            log.debug("    LUT reset fired this tick → resetting rest counter")
            reset_rest_after_lut_fire(rest_state)

    log.info("Done. Writing %d (row, cell) entries to %s", len(out_rows), output_csv)
    out_schema = {
        "sequence_number": pl.Int64,
        "cell_idx":        pl.Int64,
        "mode":            pl.Utf8,
        "soc_min":         pl.Float64,
        "soc_max":         pl.Float64,
        "soc_mid":         pl.Float64,
        "soc_hybrid":      pl.Float64,
        "time_stamp":      pl.Utf8,
        "soc_nominal":     pl.Float64,
    }
    pl.DataFrame(out_rows, schema=out_schema).write_csv(output_csv)

    log.info("Per-cell summary:")
    for i, st in enumerate(cell_states, start=1):
        log.info("  cell %2d: reset_count=%d  final_soc=%s",
                 i, st["reset_count"], st["prev_soc"])


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",          required=True,  help="Input CSV path")
    p.add_argument("--lut_csv",      required=True,  help="OCV LUT CSV path (soc, voltage columns)")
    p.add_argument("--capacity",     required=True,  type=float, help="Pack capacity Ah")
    p.add_argument("--num_cells",    required=True,  type=int)
    p.add_argument(
        "--rest_current_a", default=1.0, type=float,
        help="Pack current must stay within ±this value (A) for the rest window "
             "(default 1.0 A)",
    )
    p.add_argument(
        "--rest_min_ticks", default=5, type=int,
        help="Number of consecutive 60-s ticks within rest_current_a required "
             "before an OCV reset is allowed (default 5 = 5 min). "
             "Counter resets to zero after each LUT reset fires.",
    )
    p.add_argument(
        "--voltage_tolerance_mv", default=0.0, type=float,
        help="Optional: only allow OCV reset when the cell voltage is within "
             "this many mV of the nearest LUT point (0 = disabled, default).",
    )
    p.add_argument(
        "--voltage_delta_min_mv", default=0.0, type=float,
        help="Optional: gate OCV reset on |v_cell[N] - v_cell[N-1]| >= this "
             "many mV (0 = disabled, default).",
    )
    p.add_argument("--output_csv", default="soc_ocv_output.csv")
    p.add_argument("--log_file",   default="soc_sequential_ocv.log")
    args = p.parse_args()

    log = setup_logging(args.log_file)
    log.info("------ soc_sequential_ocv start ------")
    log.info("Args: %s", vars(args))

    if not Path(args.csv).exists():
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)
    if not Path(args.lut_csv).exists():
        log.error("OCV LUT CSV not found: %s", args.lut_csv)
        sys.exit(1)

    ocv_curve = load_ocv_lut(args.lut_csv, log)

    run_sequential(
        log=log,
        csv_path=args.csv,
        ocv_curve=ocv_curve,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        rest_current_a=args.rest_current_a,
        rest_min_ticks=args.rest_min_ticks,
        voltage_tolerance_mv=args.voltage_tolerance_mv,
        voltage_delta_min_mv=args.voltage_delta_min_mv,
        output_csv=args.output_csv,
    )
    log.info("------ soc_sequential_ocv done ------")


if __name__ == "__main__":
    main()
