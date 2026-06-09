"""
Integration tests: CSV source → engine → results.

Uses the 20-row synthetic fixture (tests/fixtures/sample_input.csv).

Scenario
--------
Rows 1–10:  current = 5.0 A (charging, above rest threshold of 1.0 A)
Rows 11–20: current = 0.1 A (at rest)

With rest_min_ticks = 5, the rest window qualifies at row 15 (5th consecutive
rest tick).  Expected mode sequence for every cell:

  rows 1–14  → UNINIT  (not at rest / not yet initialized)
  row  15    → LUT     (rest qualified; LUT fires → cell initialized)
  rows 16–19 → CC      (rest counter reset after LUT fire; new window building)
  row  20    → LUT     (second rest window qualifies; LUT fires again)
"""

from pathlib import Path

import pytest

from soc_estimation.data_sources.csv_source import CSVDataSource
from soc_estimation.engine import SOCEstimationEngine

FIXTURES = Path(__file__).parent / "fixtures"
NUM_CELLS = 4
CAPACITY_AH = 10.0
REST_CURRENT_A = 1.0
REST_MIN_TICKS = 5


@pytest.fixture
def engine(ocv_lut_path: Path) -> SOCEstimationEngine:
    source = CSVDataSource(FIXTURES / "sample_input.csv", num_cells=NUM_CELLS)
    return SOCEstimationEngine(
        data_source=source,
        num_cells=NUM_CELLS,
        capacity_ah=CAPACITY_AH,
        ocv_lut_path=ocv_lut_path,
        rest_current_a=REST_CURRENT_A,
        rest_min_ticks=REST_MIN_TICKS,
    )


@pytest.fixture
def results(engine: SOCEstimationEngine) -> list[dict]:
    return engine.process_all_records()


# ── Basic shape ───────────────────────────────────────────────────────────────


def test_result_count(results: list[dict]):
    """One row per cell per tick → 20 ticks × 4 cells = 80 rows."""
    assert len(results) == 20 * NUM_CELLS


def test_result_keys(results: list[dict]):
    expected_keys = {
        "sequence_number",
        "cell_idx",
        "mode",
        "soc_min",
        "soc_max",
        "soc_mid",
        "soc_hybrid",
        "time_stamp",
        "soc_nominal",
    }
    assert set(results[0].keys()) == expected_keys


def test_cell_indices(results: list[dict]):
    """All four cell indices appear in every tick."""
    for seq in range(1, 21):
        tick_rows = [r for r in results if r["sequence_number"] == seq]
        assert {r["cell_idx"] for r in tick_rows} == {1, 2, 3, 4}


# ── Mode sequence ─────────────────────────────────────────────────────────────


def _modes_for_cell(results: list[dict], cell_idx: int) -> list[str]:
    rows = sorted(
        (r for r in results if r["cell_idx"] == cell_idx),
        key=lambda r: r["sequence_number"],
    )
    return [r["mode"] for r in rows]


@pytest.mark.parametrize("cell_idx", [1, 2, 3, 4])
def test_mode_sequence(results: list[dict], cell_idx: int):
    """
    Expected mode sequence for each cell:
      rows 1-14  → UNINIT  (14 rows)
      row  15    → LUT
      rows 16-19 → CC      (4 rows)
      row  20    → LUT
    """
    modes = _modes_for_cell(results, cell_idx)
    assert modes[:14] == ["UNINIT"] * 14, f"cell {cell_idx}: expected UNINIT for rows 1-14"
    assert modes[14] == "LUT", f"cell {cell_idx}: expected LUT at row 15"
    assert modes[15:19] == ["CC"] * 4, f"cell {cell_idx}: expected CC for rows 16-19"
    assert modes[19] == "LUT", f"cell {cell_idx}: expected LUT at row 20"


# ── LUT reset values ──────────────────────────────────────────────────────────


def _lut_rows(results: list[dict], cell_idx: int) -> list[dict]:
    return [r for r in results if r["cell_idx"] == cell_idx and r["mode"] == "LUT"]


@pytest.mark.parametrize("cell_idx", [1, 2, 3, 4])
def test_lut_soc_within_range(results: list[dict], cell_idx: int):
    """LUT SOC estimates should be in the 60–80 % range for the fixture voltages."""
    for row in _lut_rows(results, cell_idx):
        assert row["soc_hybrid"] is not None
        assert 60.0 <= row["soc_hybrid"] <= 80.0, (
            f"cell {cell_idx} LUT SoC {row['soc_hybrid']:.2f}% out of expected range"
        )


def test_lut_fires_at_sequence_15(results: list[dict]):
    """First LUT reset happens at sequence_number = 15 for all cells."""
    first_lut: dict[int, int] = {}
    for r in results:
        if r["mode"] == "LUT":
            ci = r["cell_idx"]
            seq = r["sequence_number"]
            if ci not in first_lut or seq < first_lut[ci]:
                first_lut[ci] = seq

    for cell_idx in range(1, NUM_CELLS + 1):
        assert first_lut[cell_idx] == 15, (
            f"cell {cell_idx}: first LUT fired at seq {first_lut[cell_idx]}, expected 15"
        )


# ── CC behaviour ──────────────────────────────────────────────────────────────


def test_cc_soc_close_to_lut_soc(results: list[dict]):
    """CC SoC immediately after a LUT reset should be close to the LUT value."""
    for cell_idx in range(1, NUM_CELLS + 1):
        lut_row = next(r for r in results if r["cell_idx"] == cell_idx and r["mode"] == "LUT")
        lut_soc = lut_row["soc_hybrid"]

        # First CC row follows immediately after the LUT row
        cc_row = next(
            r
            for r in results
            if r["cell_idx"] == cell_idx
            and r["mode"] == "CC"
            and r["sequence_number"] == lut_row["sequence_number"] + 1
        )
        assert cc_row["soc_hybrid"] == pytest.approx(lut_soc, abs=2.0), (
            f"cell {cell_idx}: CC SoC {cc_row['soc_hybrid']:.2f}% too far from "
            f"LUT SoC {lut_soc:.2f}%"
        )


# ── UNINIT has no SoC ─────────────────────────────────────────────────────────


def test_uninit_has_no_soc(results: list[dict]):
    uninit_rows = [r for r in results if r["mode"] == "UNINIT"]
    assert len(uninit_rows) > 0
    for row in uninit_rows:
        assert row["soc_hybrid"] is None, (
            f"UNINIT row at seq {row['sequence_number']} has unexpected soc_hybrid"
        )


# ── CSV data source standalone ────────────────────────────────────────────────


def test_csv_source_reads_all_rows():
    source = CSVDataSource(FIXTURES / "sample_input.csv", num_cells=NUM_CELLS)
    records = list(source.get_records())
    assert len(records) == 20


def test_csv_source_delta_charge_first_row_is_none():
    """First row has no previous tick, so delta_charge must be None."""
    source = CSVDataSource(FIXTURES / "sample_input.csv", num_cells=NUM_CELLS)
    first = next(source.get_records())
    assert first.delta_charge is None


def test_csv_source_delta_charge_second_row():
    """Second row: charge_in increased by 300 C, charge_out unchanged → delta=300."""
    source = CSVDataSource(FIXTURES / "sample_input.csv", num_cells=NUM_CELLS)
    records = list(source.get_records())
    assert records[1].delta_charge == pytest.approx(300.0)
