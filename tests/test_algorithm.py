"""
Unit tests for the core OCV algorithm.

These tests cover the pure algorithm functions and stateful objects without
any file I/O.  The OCV curve used here matches tests/fixtures/ocv_lut.csv:

  soc   0 →  3000 mV
  soc  10 →  3300 mV
  soc  20 →  3500 mV
  soc  30 →  3600 mV
  soc  40 →  3680 mV
  soc  50 →  3740 mV
  soc  60 →  3790 mV
  soc  70 →  3840 mV   ← used in most tests below
  soc  80 →  3900 mV
  soc  90 →  3960 mV
  soc 100 →  4100 mV
"""

import pytest

from soc_estimation.core.ocv_algorithm import (
    CellState,
    RestWindowState,
    process_cell_tick,
    voltage_to_soc,
)

# ── voltage_to_soc ────────────────────────────────────────────────────────────


class TestVoltageToSoc:
    def test_exact_lut_point(self, ocv_curve):
        """Voltage exactly on a LUT point returns that point's SoC."""
        assert voltage_to_soc(ocv_curve, 3840.0) == pytest.approx(70.0)
        assert voltage_to_soc(ocv_curve, 3000.0) == pytest.approx(0.0)
        assert voltage_to_soc(ocv_curve, 4100.0) == pytest.approx(100.0)

    def test_interpolation_midpoint(self, ocv_curve):
        """Halfway between two LUT points returns the average SoC."""
        # Between 3840 (70%) and 3900 (80%) → midpoint = 3870 → 75%
        assert voltage_to_soc(ocv_curve, 3870.0) == pytest.approx(75.0)

    def test_interpolation_arbitrary(self, ocv_curve):
        """Linear interpolation is correct for an arbitrary voltage."""
        # Between 3600 (30%) and 3680 (40%)
        # fraction = (3640 - 3600) / (3680 - 3600) = 40/80 = 0.5
        assert voltage_to_soc(ocv_curve, 3640.0) == pytest.approx(35.0)

    def test_voltage_above_max_clamps_to_100(self, ocv_curve):
        assert voltage_to_soc(ocv_curve, 9999.0) == pytest.approx(100.0)

    def test_voltage_below_min_clamps_to_0(self, ocv_curve):
        assert voltage_to_soc(ocv_curve, 1000.0) == pytest.approx(0.0)

    def test_voltage_at_max_boundary(self, ocv_curve):
        assert voltage_to_soc(ocv_curve, 4100.0) == pytest.approx(100.0)

    def test_voltage_at_min_boundary(self, ocv_curve):
        assert voltage_to_soc(ocv_curve, 3000.0) == pytest.approx(0.0)


# ── RestWindowState ───────────────────────────────────────────────────────────


class TestRestWindowState:
    def _make(self, rest_current_a=1.0, rest_min_ticks=5) -> RestWindowState:
        return RestWindowState(rest_current_a=rest_current_a, rest_min_ticks=rest_min_ticks)

    def test_qualifies_after_min_ticks(self):
        rws = self._make(rest_current_a=1.0, rest_min_ticks=3)
        assert rws.update(0.5) is False  # tick 1
        assert rws.update(0.5) is False  # tick 2
        assert rws.update(0.5) is True  # tick 3 → qualified

    def test_does_not_qualify_before_min_ticks(self):
        rws = self._make(rest_min_ticks=5)
        for _ in range(4):
            assert rws.update(0.2) is False
        assert rws.update(0.2) is True  # 5th tick qualifies

    def test_high_current_resets_counter(self):
        rws = self._make(rest_current_a=1.0, rest_min_ticks=3)
        rws.update(0.5)
        rws.update(0.5)
        rws.update(5.0)  # exceeds threshold → reset
        assert rws.consecutive_ticks == 0

    def test_negative_current_also_resets(self):
        """Discharging at high rate also breaks the rest window."""
        rws = self._make(rest_current_a=1.0, rest_min_ticks=3)
        rws.update(0.5)
        rws.update(-5.0)  # discharging hard
        assert rws.consecutive_ticks == 0

    def test_none_current_resets_counter(self):
        rws = self._make(rest_min_ticks=3)
        rws.update(0.2)
        rws.update(None)  # unknown current → treat as not at rest
        assert rws.consecutive_ticks == 0

    def test_reset_after_lut_fire_requires_fresh_window(self):
        rws = self._make(rest_current_a=1.0, rest_min_ticks=3)
        for _ in range(3):
            rws.update(0.2)
        assert rws.consecutive_ticks == 3

        rws.reset()  # simulate LUT fire
        assert rws.consecutive_ticks == 0
        assert rws.update(0.2) is False  # needs fresh 3 ticks


# ── process_cell_tick ─────────────────────────────────────────────────────────


class TestProcessCellTick:
    """End-to-end tests for the per-cell decision function."""

    def _state(self) -> CellState:
        return CellState()

    def test_uninit_when_not_at_rest(self, ocv_curve):
        """Cell stays UNINIT when pack is not at rest and never initialized."""
        state = self._state()
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.840,
            delta_charge=100.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=False,
        )
        assert result.mode == "UNINIT"
        assert result.soc_hybrid is None
        assert not state.initialized

    def test_lut_fires_when_rest_qualified(self, ocv_curve):
        """LUT reset fires immediately when rest window is qualified."""
        state = self._state()
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.840,  # → 70% SOC
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
        )
        assert result.mode == "LUT"
        assert result.soc_hybrid == pytest.approx(70.0)
        assert state.initialized
        assert state.reset_count == 1
        assert state.prev_soc == pytest.approx(70.0)

    def test_cc_after_lut_reset(self, ocv_curve):
        """After a LUT reset, CC integrates charge correctly."""
        state = self._state()

        # Initialize via LUT
        process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.840,
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
        )
        assert state.prev_soc == pytest.approx(70.0)

        # Next tick: not at rest → CC
        # delta_charge = 360 C → d_soc = 360 / (10 * 3600) * 100 = 1 %
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.845,
            delta_charge=360.0,  # +1 % SOC equivalent for 10 Ah pack
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=False,
        )
        assert result.mode == "CC"
        assert result.soc_hybrid == pytest.approx(71.0)

    def test_skip_on_invalid_voltage(self, ocv_curve):
        """SKIP is returned for missing or zero voltage."""
        state = self._state()
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=None,
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
        )
        assert result.mode == "SKIP"

        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=0.0,
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
        )
        assert result.mode == "SKIP"

    def test_voltage_tolerance_gate_blocks_lut(self, ocv_curve):
        """LUT reset is blocked when voltage is far from all LUT points."""
        state = self._state()
        # 3820 mV is 20 mV from the nearest LUT point (3840); tolerance=10 → blocked
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.820,
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
            voltage_tolerance_mv=10.0,
        )
        assert result.mode == "UNINIT"

    def test_voltage_tolerance_gate_allows_lut(self, ocv_curve):
        """LUT reset fires when voltage is within the tolerance band."""
        state = self._state()
        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.838,  # 2 mV from 3840 LUT point
            delta_charge=0.0,
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=True,
            voltage_tolerance_mv=5.0,
        )
        assert result.mode == "LUT"

    def test_soc_clamped_to_0_100(self, ocv_curve):
        """Extreme CC drift is clamped to [0, 100]."""
        state = self._state()
        state.initialized = True
        state.prev_soc = 99.0

        result = process_cell_tick(
            cell_idx=1,
            cell_voltage_v=3.840,
            delta_charge=100_000.0,  # enormous charge → would overflow
            cell_state=state,
            ocv_curve=ocv_curve,
            capacity_ah=10.0,
            rest_qualified=False,
        )
        assert result.mode == "CC"
        assert result.soc_hybrid == pytest.approx(100.0)
