"""
Core OCV (Open-Circuit Voltage) sequential state-of-charge estimation algorithm.

This module implements the per-cell, per-tick SoC estimation logic using an
OCV lookup table with intelligent rest-window gating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

SECONDS_PER_TICK = 60.0  # Athena cadence (60 seconds)


# ── Cell state ────────────────────────────────────────────────────────────────
@dataclass
class CellState:
    """State tracking for a single cell across ticks."""

    prev_soc: float | None = None  # Last SoC estimate (None until first LUT reset)
    initialized: bool = False  # Has this cell been initialized?
    reset_count: int = 0  # Number of LUT resets applied
    prev_voltage_mv: float | None = None  # Cell voltage last tick (mV)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "prev_soc": self.prev_soc,
            "initialized": self.initialized,
            "reset_count": self.reset_count,
            "prev_voltage_mv": self.prev_voltage_mv,
        }


# ── Rest window state ─────────────────────────────────────────────────────────
@dataclass
class RestWindowState:
    """Tracks pack-level rest window for LUT resets."""

    rest_current_a: float  # Pack current threshold (A)
    rest_min_ticks: int  # Consecutive ticks required
    consecutive_ticks: int = 0  # Current consecutive rest ticks

    def update(self, current_a: float | None) -> bool:
        """
        Update rest window state with new current reading.

        Args:
            current_a: Pack current (None = out-of-rest)

        Returns:
            True if rest window is now qualified
        """
        if current_a is None or abs(float(current_a)) > self.rest_current_a:
            self.consecutive_ticks = 0
            return False

        self.consecutive_ticks += 1
        return self.consecutive_ticks >= self.rest_min_ticks

    def reset(self) -> None:
        """Reset counter after LUT fire (new rest window required)."""
        self.consecutive_ticks = 0


# ── OCV interpolation ─────────────────────────────────────────────────────────
def voltage_to_soc(
    ocv_curve: list[tuple[float, float]],
    voltage_mv: float,
) -> float:
    """
    Linear interpolation: voltage (mV) → SoC (%).

    OCV curve is a list of (voltage_mv, soc_pct) sorted by voltage.
    - Voltage above highest LUT point → 100 %
    - Voltage below lowest LUT point → 0 %
    - Otherwise linear interpolation between bracketing points

    Args:
        ocv_curve: List of (voltage_mv, soc_pct) tuples sorted by voltage
        voltage_mv: Cell voltage in mV

    Returns:
        Estimated SoC percentage (0–100)
    """
    v_min, soc_at_vmin = ocv_curve[0]
    v_max, soc_at_vmax = ocv_curve[-1]

    if voltage_mv >= v_max:
        return 100.0
    if voltage_mv <= v_min:
        return 0.0

    # Binary search for lower bracket
    lo, hi = 0, len(ocv_curve) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ocv_curve[mid][0] <= voltage_mv:
            lo = mid
        else:
            hi = mid

    v_lo, soc_lo = ocv_curve[lo]
    v_hi, soc_hi = ocv_curve[hi]

    # Guard against duplicate voltage entries
    if v_hi == v_lo:
        return float(soc_lo)

    fraction = (voltage_mv - v_lo) / (v_hi - v_lo)
    soc = soc_lo + fraction * (soc_hi - soc_lo)
    return max(0.0, min(100.0, soc))


# ── Per-tick cell result ──────────────────────────────────────────────────────
@dataclass
class CellTickResult:
    """Result of processing one cell for one tick."""

    sequence_number: int
    cell_idx: int
    mode: str  # "LUT", "CC", "UNINIT", or "SKIP"
    soc_min: float | None = None  # Min SoC in bracket
    soc_max: float | None = None  # Max SoC in bracket
    soc_mid: float | None = None  # Mid SoC (usually same as soc_max)
    soc_hybrid: float | None = None  # Final hybrid SoC (LUT or CC)
    time_stamp: str | None = None
    soc_nominal: float | None = None  # Reported BMS SoC (for reference)

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV output."""
        return {
            "sequence_number": self.sequence_number,
            "cell_idx": self.cell_idx,
            "mode": self.mode,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_mid": self.soc_mid,
            "soc_hybrid": self.soc_hybrid,
            "time_stamp": self.time_stamp,
            "soc_nominal": self.soc_nominal,
        }


# ── Per-cell, per-tick processing ─────────────────────────────────────────────
def process_cell_tick(
    cell_idx: int,
    cell_voltage_v: float | None,
    delta_charge: float | None,
    cell_state: CellState,
    ocv_curve: list[tuple[float, float]],
    capacity_ah: float,
    rest_qualified: bool,
    voltage_tolerance_mv: float = 0.0,
    voltage_delta_min_mv: float = 0.0,
    logger: logging.Logger | None = None,
) -> CellTickResult:
    """
    Process one cell for one tick using OCV LUT with rest gating.

    Args:
        cell_idx: Cell index (1-based)
        cell_voltage_v: Cell voltage in volts (None = invalid)
        delta_charge: Charge delta since last tick in Coulombs (None = unknown)
        cell_state: Cell state object
        ocv_curve: OCV lookup table (voltage_mv, soc_pct) sorted by voltage
        capacity_ah: Cell/pack capacity in Ah
        rest_qualified: Pack rest window currently qualified?
        voltage_tolerance_mv: Optional tolerance gate (0 = disabled)
        voltage_delta_min_mv: Optional voltage-delta gate (0 = disabled)
        logger: Optional logger

    Returns:
        CellTickResult with SoC estimation for this tick
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Step 1: Validate inputs
    if cell_voltage_v is None or cell_voltage_v <= 0:
        return CellTickResult(
            sequence_number=0,
            cell_idx=cell_idx,
            mode="SKIP",
            soc_hybrid=cell_state.prev_soc,
        )

    cell_voltage_mv = cell_voltage_v * 1000.0

    # Step 2: Rest window gate
    if not rest_qualified:
        # Not at rest long enough; fall back to coulomb counting
        cell_state.prev_voltage_mv = cell_voltage_mv
        return _run_cc_or_uninit(cell_idx, cell_state, cell_voltage_mv, delta_charge, capacity_ah)

    # Step 3: Optional voltage-delta gate
    prev_v = cell_state.prev_voltage_mv
    cell_state.prev_voltage_mv = cell_voltage_mv

    if voltage_delta_min_mv > 0.0:
        if prev_v is None:
            return _run_cc_or_uninit(
                cell_idx, cell_state, cell_voltage_mv, delta_charge, capacity_ah
            )
        v_delta = abs(cell_voltage_mv - prev_v)
        if v_delta < voltage_delta_min_mv:
            return _run_cc_or_uninit(
                cell_idx, cell_state, cell_voltage_mv, delta_charge, capacity_ah
            )

    # Step 4: OCV interpolation
    soc_interp = voltage_to_soc(ocv_curve, cell_voltage_mv)

    # Step 5: Optional voltage-tolerance gate
    if voltage_tolerance_mv > 0.0:
        nearest_dist = min(abs(cell_voltage_mv - v) for v, _ in ocv_curve)
        if nearest_dist > voltage_tolerance_mv:
            return _run_cc_or_uninit(
                cell_idx, cell_state, cell_voltage_mv, delta_charge, capacity_ah
            )

    # Step 6: LUT reset confirmed
    soc_interp = max(0.0, min(100.0, soc_interp))
    cell_state.prev_soc = soc_interp
    cell_state.initialized = True
    cell_state.reset_count += 1

    return CellTickResult(
        sequence_number=0,
        cell_idx=cell_idx,
        mode="LUT",
        soc_min=soc_interp,
        soc_max=soc_interp,
        soc_mid=soc_interp,
        soc_hybrid=soc_interp,
    )


def _run_cc_or_uninit(
    cell_idx: int,
    cell_state: CellState,
    cell_voltage_mv: float,
    delta_charge: float | None,
    capacity_ah: float,
) -> CellTickResult:
    """
    Fall back to coulomb counting or return uninitialized.

    If cell is not yet initialized, return UNINIT (no SoC).
    Otherwise, integrate coulomb counting on top of last known SoC.
    """
    if not cell_state.initialized:
        return CellTickResult(
            sequence_number=0,
            cell_idx=cell_idx,
            mode="UNINIT",
            soc_hybrid=None,
        )

    dc = 0.0 if delta_charge is None else float(delta_charge)
    d_soc = dc / (capacity_ah * 3600.0) * 100.0
    new_soc = cell_state.prev_soc + d_soc
    new_soc_clamped = max(0.0, min(100.0, new_soc))
    cell_state.prev_soc = new_soc_clamped

    return CellTickResult(
        sequence_number=0,
        cell_idx=cell_idx,
        mode="CC",
        soc_hybrid=new_soc_clamped,
    )
