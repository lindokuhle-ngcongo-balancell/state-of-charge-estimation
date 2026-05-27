# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "numpy"]
# ///
"""
soc_sequential_ocv.py
─────────────────────
Sequential, row-by-row SoC estimator using OCV and constant-current LUTs.

LUT selection per tick
──────────────────────
Three mutually exclusive steady-state modes, checked in order:

  1. REST  — |current| <= rest_current_a  for >= rest_min_ticks consecutive
             ticks  →  OCV LUT  (counter resets to 0 after each reset fires)

  2. CHARGE — current has stayed within +-current_lut_tol_a of the SAME
              charging LUT current for >= rest_min_ticks consecutive ticks
              →  nearest charging LUT  (counter does NOT reset; fires every
              tick as long as current stays steady)

  3. DISCHARGE — same as CHARGE but for discharging LUTs (current negative
                 or matched to a discharge curve)

  4. CC   — none of the above; integrate delta_charge into previous SoC

Rest counter resets to zero after each OCV LUT fire.
Charge/discharge counters reset only when current drifts outside the
tolerance band of their matched LUT current.

LUT files
─────────
OCV LUT CSV:             columns  soc, voltage   (voltage in mV)
Charge/discharge CSVs:   columns  voltage, soc   (voltage in V → converted to mV)
                         filenames must contain  Charging_Current_<N>A  or
                         Discharging_Current_<N>A  so the current can be parsed.
                         The 9A charging file is intentionally excluded (too few
                         points) — just don't put it in the lut_dir folder.

Folder layout expected
──────────────────────
  data/csv_lut/
      lut_ocv.csv
      Single_Cell_277Ah_AT_Constant_Charging_Current_2A.csv
      Single_Cell_277Ah_AT_Constant_Charging_Current_4A.csv
      ...
      Single_Cell_277Ah_AT_Constant_Discharging_Current_2A.csv
      ...

Usage
─────
uv run soc_sequential_ocv.py \\
    --csv              data/raw.csv \\
    --lut_dir          data/csv_lut \\
    --ocv_lut_file     lut_ocv.csv \\
    --capacity         277 \\
    --num_cells        32 \\
    --output_csv       soc_ocv_output.csv \\
    --log_file         soc_sequential_ocv.log

Optional flags:
    --rest_current_a        1.0    ±A threshold for OCV rest window
    --rest_min_ticks        5      consecutive ticks for any steady-state window
    --current_lut_tol_a     2.5    ±A tolerance for matching to a charge/discharge LUT
    --voltage_tolerance_mv  0.0    reject LUT reset if cell V is >N mV from nearest
                                   LUT point (0 = disabled)
    --voltage_delta_min_mv  0.0    gate LUT reset on |v[N]-v[N-1]| >= N mV
                                   (0 = disabled)
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import re
import sys
from pathlib import Path

import polars as pl


# ── Constants ────────────────────────────────────────────────────────────────
SECONDS_PER_TICK = 60.0


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> logging.Logger:
    """File gets DEBUG; stderr gets INFO."""
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


# ── Curve type ───────────────────────────────────────────────────────────────
# A curve is a list of (voltage_mv, soc_pct) sorted ascending by voltage.
Curve = list[tuple[float, float]]


# ── LUT loaders ──────────────────────────────────────────────────────────────
def _sort_and_log(
    pairs: list[tuple[float, float]],
    label: str,
    log: logging.Logger,
) -> Curve:
    """Sort (voltage_mv, soc) pairs by voltage and log a summary."""
    pairs.sort(key=lambda x: x[0])
    log.info(
        "  %-45s  %4d pts  V=[%6.0f,%6.0f] mV  SoC=[%5.1f,%5.1f]%%",
        label, len(pairs),
        pairs[0][0], pairs[-1][0],
        pairs[0][1], pairs[-1][1],
    )
    return pairs


def load_ocv_lut(csv_path: Path, log: logging.Logger) -> Curve:
    """
    OCV LUT:  columns  soc, voltage  (voltage already in mV).
    Returns list of (voltage_mv, soc) sorted ascending by voltage.
    """
    log.info("Loading OCV LUT: %s", csv_path)
    df = pl.read_csv(str(csv_path))
    missing = {"soc", "voltage"} - set(df.columns)
    if missing:
        raise ValueError(f"OCV LUT missing columns: {sorted(missing)}")
    pairs = [(float(r["voltage"]), float(r["soc"])) for r in df.iter_rows(named=True)]
    return _sort_and_log(pairs, csv_path.name, log)


def load_current_lut(csv_path: Path, log: logging.Logger) -> Curve:
    """
    Charge/discharge LUT:  columns  voltage, soc  (voltage in V → converted to mV).
    Returns list of (voltage_mv, soc) sorted ascending by voltage.
    """
    df = pl.read_csv(str(csv_path))
    missing = {"voltage", "soc"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {sorted(missing)}")
    pairs = [
        (float(r["voltage"]) * 1000.0, float(r["soc"]))
        for r in df.iter_rows(named=True)
    ]
    return _sort_and_log(pairs, csv_path.name, log)


def load_all_luts(
    lut_dir: Path,
    ocv_filename: str,
    log: logging.Logger,
) -> tuple[Curve, dict[float, Curve], dict[float, Curve]]:
    """
    Load OCV LUT plus all charge/discharge LUTs found in lut_dir.

    Returns:
        ocv_curve        — Curve
        charge_luts      — {current_a: Curve}   current_a > 0
        discharge_luts   — {current_a: Curve}   current_a > 0  (magnitude)
    """
    ocv_curve = load_ocv_lut(lut_dir / ocv_filename, log)

    charge_luts:    dict[float, Curve] = {}
    discharge_luts: dict[float, Curve] = {}

    pattern = re.compile(r"(Charging|Discharging)_Current_(\d+)A", re.IGNORECASE)

    for csv_path in sorted(lut_dir.glob("*.csv")):
        if csv_path.name == ocv_filename:
            continue
        m = pattern.search(csv_path.name)
        if not m:
            log.warning("Skipping unrecognised LUT file: %s", csv_path.name)
            continue
        direction = m.group(1).lower()   # "charging" or "discharging"
        current_a = float(m.group(2))
        curve = load_current_lut(csv_path, log)
        if direction == "charging":
            charge_luts[current_a] = curve
        else:
            discharge_luts[current_a] = curve

    log.info(
        "LUTs loaded — OCV: 1  Charging: %d %s  Discharging: %d %s",
        len(charge_luts),    sorted(charge_luts.keys()),
        len(discharge_luts), sorted(discharge_luts.keys()),
    )
    return ocv_curve, charge_luts, discharge_luts


# ── Interpolation ─────────────────────────────────────────────────────────────
def interpolate_soc(curve: Curve, voltage_mv: float) -> float:
    """
    Linear interpolation: voltage_mv → SoC (%).
    Clamps to 0 % below the curve and 100 % above it.
    """
    if voltage_mv >= curve[-1][0]:
        return 100.0
    if voltage_mv <= curve[0][0]:
        return 0.0
    lo, hi = 0, len(curve) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if curve[mid][0] <= voltage_mv:
            lo = mid
        else:
            hi = mid
    v_lo, soc_lo = curve[lo]
    v_hi, soc_hi = curve[hi]
    if v_hi == v_lo:
        return float(soc_lo)
    fraction = (voltage_mv - v_lo) / (v_hi - v_lo)
    return max(0.0, min(100.0, soc_lo + fraction * (soc_hi - soc_lo)))


def nearest_lut_distance(curve: Curve, voltage_mv: float) -> float:
    """Minimum absolute distance from voltage_mv to any point in the curve."""
    return min(abs(voltage_mv - v) for v, _ in curve)


# ── Current-LUT matcher ───────────────────────────────────────────────────────
def find_nearest_lut(
    lut_dict: dict[float, Curve],
    current_magnitude: float,
    tolerance_a: float,
) -> tuple[float, Curve] | None:
    """
    Find the LUT whose nominal current is closest to current_magnitude,
    but only if |nominal - current_magnitude| <= tolerance_a.

    Returns (nominal_current_a, curve) or None.
    """
    best_current: float | None = None
    best_dist = float("inf")
    for nominal in lut_dict:
        dist = abs(nominal - current_magnitude)
        if dist <= tolerance_a and dist < best_dist:
            best_dist = dist
            best_current = nominal
    if best_current is None:
        return None
    return best_current, lut_dict[best_current]


# ── Rest / steady-state window tracker ───────────────────────────────────────
def make_window_state() -> dict:
    """
    Tracks consecutive ticks for one steady-state condition.
    For OCV (rest): caller resets after each LUT fire.
    For charge/discharge: caller resets only when current drifts out of band.
    """
    return {"consecutive_ticks": 0}


def update_window(state: dict, condition_met: bool, min_ticks: int) -> bool:
    """Increment counter if condition met, else reset. Returns qualified bool."""
    if condition_met:
        state["consecutive_ticks"] += 1
    else:
        state["consecutive_ticks"] = 0
    return state["consecutive_ticks"] >= min_ticks


# ── Per-cell state ────────────────────────────────────────────────────────────
def make_cell_state() -> dict:
    return {
        "prev_soc":        None,
        "initialized":     False,
        "reset_count":     0,
        "prev_voltage_mv": None,
    }


# ── CC / UNINIT fallback ──────────────────────────────────────────────────────
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
        log.debug("    initialized=False → UNINIT  (%s)", reason)
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "UNINIT",
            "lut_current_a": None,
            "soc_min": None, "soc_max": None, "soc_mid": None, "soc_hybrid": None,
        }
    dc = 0.0 if delta_charge is None else float(delta_charge)
    d_soc = dc / (capacity_ah * 3600.0) * 100.0
    new_soc = max(0.0, min(100.0, state["prev_soc"] + d_soc))
    log.debug(
        "    CC  (%s)  prev=%.4f%%  dQ=%.4fC  dSoC=%.6f%%  new=%.4f%%",
        reason, state["prev_soc"], dc, d_soc, new_soc,
    )
    state["prev_soc"] = new_soc
    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": "CC",
        "lut_current_a": None,
        "soc_min": None, "soc_max": None, "soc_mid": None, "soc_hybrid": new_soc,
    }


# ── Core per-(tick, cell) logic ───────────────────────────────────────────────
def process_one_cell_tick(
    log: logging.Logger,
    seq: int,
    time_stamp,
    cell_idx: int,
    cell_voltage_v: float | None,
    delta_charge: float | None,
    state: dict,
    # which LUT (if any) the pack-level window tracker selected this tick
    active_curve: Curve | None,
    active_mode: str,           # "OCV", "CHARGE", "DISCHARGE", or "CC"
    active_lut_current_a: float | None,
    capacity_ah: float,
    voltage_tolerance_mv: float,
    voltage_delta_min_mv: float,
) -> dict:
    log.debug("─" * 70)
    log.debug("ROW seq=%d  time=%s  cell=%d  pack_mode=%s", seq, time_stamp, cell_idx, active_mode)
    log.debug("─" * 70)

    # ── Validate ─────────────────────────────────────────────────────────────
    if cell_voltage_v is None or cell_voltage_v <= 0:
        log.debug("    cell_voltage invalid → SKIP")
        return {
            "sequence_number": seq, "cell_idx": cell_idx,
            "mode": "SKIP", "lut_current_a": None,
            "soc_min": None, "soc_max": None, "soc_mid": None,
            "soc_hybrid": state["prev_soc"],
        }

    cell_voltage_mv = cell_voltage_v * 1000.0
    log.debug("    cell_voltage_mV = %.2f", cell_voltage_mv)

    # ── No LUT selected at pack level → CC ───────────────────────────────────
    if active_curve is None:
        state["prev_voltage_mv"] = cell_voltage_mv
        return _run_cc_or_uninit(
            log, seq, cell_idx, delta_charge, state, capacity_ah,
            reason=f"pack mode={active_mode}, no LUT active",
        )

    # ── Optional voltage-delta gate ───────────────────────────────────────────
    prev_v = state["prev_voltage_mv"]
    state["prev_voltage_mv"] = cell_voltage_mv

    if voltage_delta_min_mv > 0.0:
        log.debug("[VDELTA] threshold=%.2f mV", voltage_delta_min_mv)
        if prev_v is None:
            log.debug("    no prev voltage (first tick) → CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="first tick, no prev voltage",
            )
        v_delta = abs(cell_voltage_mv - prev_v)
        log.debug("    |v_delta|=%.2f mV", v_delta)
        if v_delta < voltage_delta_min_mv:
            log.debug("    below threshold → CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="voltage delta below threshold",
            )

    # ── Optional voltage-tolerance gate ──────────────────────────────────────
    if voltage_tolerance_mv > 0.0:
        dist = nearest_lut_distance(active_curve, cell_voltage_mv)
        log.debug("[VTOL] nearest LUT point dist=%.2f mV  threshold=%.2f mV", dist, voltage_tolerance_mv)
        if dist > voltage_tolerance_mv:
            log.debug("    outside tolerance → CC/UNINIT")
            return _run_cc_or_uninit(
                log, seq, cell_idx, delta_charge, state, capacity_ah,
                reason="cell voltage outside tolerance of nearest LUT point",
            )

    # ── LUT reset ─────────────────────────────────────────────────────────────
    soc = interpolate_soc(active_curve, cell_voltage_mv)
    log.debug("[LUT RESET]  mode=%s  lut_current=%.1fA  V=%.2f mV  SoC=%.4f%%",
              active_mode,
              active_lut_current_a if active_lut_current_a else 0.0,
              cell_voltage_mv, soc)

    state["prev_soc"] = soc
    state["initialized"] = True
    state["reset_count"] += 1

    return {
        "sequence_number": seq, "cell_idx": cell_idx,
        "mode": active_mode,
        "lut_current_a": active_lut_current_a,
        "soc_min": soc, "soc_max": soc, "soc_mid": soc, "soc_hybrid": soc,
    }


# ── Pack-level window state ───────────────────────────────────────────────────
def make_pack_windows() -> dict:
    return {
        "rest":      make_window_state(),
        "charge":    make_window_state(),
        "discharge": make_window_state(),
        # track which LUT the charge/discharge windows are locked onto
        "charge_locked_a":    None,
        "discharge_locked_a": None,
    }


def resolve_pack_mode(
    windows: dict,
    raw_current: float | None,
    rest_current_a: float,
    rest_min_ticks: int,
    current_lut_tol_a: float,
    charge_luts: dict[float, Curve],
    discharge_luts: dict[float, Curve],
    log: logging.Logger,
) -> tuple[str, Curve | None, float | None]:
    """
    Evaluate all three windows for this tick and return:
        (mode_str, active_curve_or_None, lut_nominal_current_or_None)

    Mode priority (all mutually exclusive in practice, but handled cleanly):
        REST > CHARGE > DISCHARGE > CC
    """
    if raw_current is None:
        # reset all windows
        for key in ("rest", "charge", "discharge"):
            windows[key]["consecutive_ticks"] = 0
        windows["charge_locked_a"] = None
        windows["discharge_locked_a"] = None
        return "CC", None, None

    current = float(raw_current)
    current_mag = abs(current)

    # ── REST window ───────────────────────────────────────────────────────────
    rest_ok = update_window(windows["rest"], current_mag <= rest_current_a, rest_min_ticks)

    # ── CHARGE window ─────────────────────────────────────────────────────────
    # current > 0 = charging convention; find nearest charging LUT
    charge_match = None
    if current > 0:
        charge_match = find_nearest_lut(charge_luts, current_mag, current_lut_tol_a)

    if charge_match is not None:
        nominal_a = charge_match[0]
        # If we were locked to a different nominal, reset
        if windows["charge_locked_a"] != nominal_a:
            windows["charge"]["consecutive_ticks"] = 0
            windows["charge_locked_a"] = nominal_a
        charge_ok = update_window(windows["charge"], True, rest_min_ticks)
    else:
        windows["charge"]["consecutive_ticks"] = 0
        windows["charge_locked_a"] = None
        charge_ok = False

    # ── DISCHARGE window ──────────────────────────────────────────────────────
    discharge_match = None
    if current < 0:
        discharge_match = find_nearest_lut(discharge_luts, current_mag, current_lut_tol_a)

    if discharge_match is not None:
        nominal_a = discharge_match[0]
        if windows["discharge_locked_a"] != nominal_a:
            windows["discharge"]["consecutive_ticks"] = 0
            windows["discharge_locked_a"] = nominal_a
        discharge_ok = update_window(windows["discharge"], True, rest_min_ticks)
    else:
        windows["discharge"]["consecutive_ticks"] = 0
        windows["discharge_locked_a"] = None
        discharge_ok = False

    log.debug(
        "[PACK WINDOW]  I=%.3fA  rest=%d/%d(%s)  chg=%d/%d(%s)  dis=%d/%d(%s)",
        current,
        windows["rest"]["consecutive_ticks"],      rest_min_ticks,      rest_ok,
        windows["charge"]["consecutive_ticks"],    rest_min_ticks,      charge_ok,
        windows["discharge"]["consecutive_ticks"], rest_min_ticks,      discharge_ok,
    )

    # ── Select active mode (priority: REST > CHARGE > DISCHARGE > CC) ────────
    if rest_ok:
        return "OCV", None, None   # curve resolved per-cell from ocv_curve arg
    if charge_ok and charge_match:
        return "CHARGE", charge_match[1], charge_match[0]
    if discharge_ok and discharge_match:
        return "DISCHARGE", discharge_match[1], discharge_match[0]
    return "CC", None, None


# ── Outer driver ──────────────────────────────────────────────────────────────
def run_sequential(
    log: logging.Logger,
    csv_path: str,
    ocv_curve: Curve,
    charge_luts: dict[float, Curve],
    discharge_luts: dict[float, Curve],
    num_cells: int,
    capacity_ah: float,
    rest_current_a: float,
    rest_min_ticks: int,
    current_lut_tol_a: float,
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

    cell_states  = [make_cell_state() for _ in range(num_cells)]
    pack_windows = make_pack_windows()
    log.info("Initialised state for %d cells", num_cells)
    log.info(
        "Rest window: ±%.2f A for %d ticks (%.0f min) — resets after OCV fire",
        rest_current_a, rest_min_ticks, rest_min_ticks * SECONDS_PER_TICK / 60.0,
    )
    log.info(
        "Charge/discharge window: ±%.2f A tolerance, %d ticks — does NOT reset after fire",
        current_lut_tol_a, rest_min_ticks,
    )

    out_rows: list[dict] = []
    rows = df.to_dicts()
    total = len(rows)
    log_every = max(1, total // 20)

    for row_idx, row in enumerate(rows):
        if row_idx % log_every == 0:
            log.info("Processing row %d / %d  (seq=%s)", row_idx, total, row["sequence_number"])

        seq        = row["sequence_number"]
        ts         = row["time_stamp"]
        raw_current = row["current"]

        # ── Pack-level mode resolution (once per tick) ────────────────────────
        active_mode, active_curve, active_lut_current_a = resolve_pack_mode(
            windows=pack_windows,
            raw_current=raw_current,
            rest_current_a=rest_current_a,
            rest_min_ticks=rest_min_ticks,
            current_lut_tol_a=current_lut_tol_a,
            charge_luts=charge_luts,
            discharge_luts=discharge_luts,
            log=log,
        )

        # For OCV mode the curve is the shared ocv_curve
        curve_this_tick = ocv_curve if active_mode == "OCV" else active_curve
        lut_current_this_tick = 0.0 if active_mode == "OCV" else active_lut_current_a

        log.debug("=" * 70)
        log.debug(
            "TICK %d/%d  seq=%s  I=%.3fA  mode=%s  lut_A=%s",
            row_idx, total, seq,
            float(raw_current) if raw_current is not None else float("nan"),
            active_mode,
            f"{lut_current_this_tick:.1f}" if active_mode != "CC" else "—",
        )
        log.debug("=" * 70)

        any_ocv_reset = False

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
                active_curve=curve_this_tick,
                active_mode=active_mode,
                active_lut_current_a=lut_current_this_tick,
                capacity_ah=capacity_ah,
                voltage_tolerance_mv=voltage_tolerance_mv,
                voltage_delta_min_mv=voltage_delta_min_mv,
            )
            if result["mode"] == "OCV":
                any_ocv_reset = True

            result["time_stamp"]  = str(ts)
            result["soc_nominal"] = row["soc"]
            out_rows.append(result)

        # OCV rest counter resets after each fire; charge/discharge do not
        if any_ocv_reset:
            log.debug("OCV reset fired → resetting rest window counter")
            pack_windows["rest"]["consecutive_ticks"] = 0

    log.info("Done. Writing %d entries to %s", len(out_rows), output_csv)

    out_schema = {
        "sequence_number": pl.Int64,
        "cell_idx":        pl.Int64,
        "mode":            pl.Utf8,
        "lut_current_a":   pl.Float64,
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
        log.info("  cell %2d: resets=%d  final_soc=%s", i, st["reset_count"], st["prev_soc"])


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",         required=True,  help="Input CSV path")
    p.add_argument("--lut_dir",     required=True,  help="Folder containing all LUT CSVs")
    p.add_argument("--ocv_lut_file", default="lut_ocv.csv",
                   help="Filename of OCV LUT inside lut_dir (default: lut_ocv.csv)")
    p.add_argument("--capacity",    required=True,  type=float, help="Pack capacity Ah")
    p.add_argument("--num_cells",   required=True,  type=int)
    p.add_argument(
        "--rest_current_a", default=1.0, type=float,
        help="±A threshold for OCV rest window (default 1.0 A)",
    )
    p.add_argument(
        "--rest_min_ticks", default=5, type=int,
        help="Consecutive 60-s ticks required for any steady-state window (default 5 = 5 min)",
    )
    p.add_argument(
        "--current_lut_tol_a", default=2.5, type=float,
        help="±A tolerance for matching pack current to a charge/discharge LUT (default 2.5 A)",
    )
    p.add_argument(
        "--voltage_tolerance_mv", default=0.0, type=float,
        help="Reject LUT reset if cell V is >N mV from nearest LUT point (0 = disabled)",
    )
    p.add_argument(
        "--voltage_delta_min_mv", default=0.0, type=float,
        help="Gate LUT reset on |v[N]-v[N-1]| >= N mV (0 = disabled)",
    )
    p.add_argument("--output_csv", default="soc_ocv_output.csv")
    p.add_argument("--log_file",   default="soc_sequential_ocv.log")
    args = p.parse_args()

    log = setup_logging(args.log_file)
    log.info("------ soc_sequential_ocv start ------")
    log.info("Args: %s", vars(args))

    lut_dir = Path(args.lut_dir)
    if not lut_dir.is_dir():
        log.error("lut_dir not found: %s", lut_dir)
        sys.exit(1)
    if not Path(args.csv).exists():
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)

    ocv_curve, charge_luts, discharge_luts = load_all_luts(lut_dir, args.ocv_lut_file, log)

    run_sequential(
        log=log,
        csv_path=args.csv,
        ocv_curve=ocv_curve,
        charge_luts=charge_luts,
        discharge_luts=discharge_luts,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        rest_current_a=args.rest_current_a,
        rest_min_ticks=args.rest_min_ticks,
        current_lut_tol_a=args.current_lut_tol_a,
        voltage_tolerance_mv=args.voltage_tolerance_mv,
        voltage_delta_min_mv=args.voltage_delta_min_mv,
        output_csv=args.output_csv,
    )
    log.info("------ soc_sequential_ocv done ------")


if __name__ == "__main__":
    main()
