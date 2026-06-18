"""
Tests for the per-cell SoH estimator (soc_estimation/soh_estimator.py).

Uses the shipped parquet LUT plus a synthetic raw frame (rest-at-top → heavy
discharge) so the q_top reference and the shared estimator both have something
to chew on. We assert structure and gating behaviour, not exact SoH values.
"""

from pathlib import Path

import polars as pl
import pytest

from soc_estimation import soh_estimator as soh
from soc_estimation.bem_estimator import load_lut, make_lut_arrays

REPO_ROOT = Path(__file__).resolve().parents[1]
LUT_PATH = (
    REPO_ROOT
    / "data"
    / "lut"
    / "voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
)

NUM_CELLS = 4
CAPACITY_AH = 277.0


@pytest.fixture(scope="module")
def lut():
    if not LUT_PATH.exists():
        pytest.skip(f"LUT parquet not found at {LUT_PATH}")
    return make_lut_arrays(load_lut(str(LUT_PATH)))


def _synthetic_raw(n: int = 400, ncells: int = NUM_CELLS) -> pl.DataFrame:
    """Rest at top for 30 ticks, then a steady heavy discharge."""
    charge_in = [100_000.0] * n
    charge_out, co = [], 50_000.0
    cellv = []
    current = []
    for i in range(n):
        if i >= 30:
            co += 7_000.0  # coulombs/min ≈ heavy discharge
        charge_out.append(co)
        v = 3.60 if i < 30 else max(3.0, 3.60 - (i - 30) * 0.0015)
        cellv.append([v] * ncells)
        current.append(0.0 if i < 30 else -116.0)
    return pl.DataFrame(
        {
            "device_serial": ["BE-TEST"] * n,
            "sequence_number": list(range(n)),
            "time_stamp": [f"2025-01-01T00:{i // 60:02d}:{i % 60:02d}" for i in range(n)],
            "current": current,
            "charge_in": charge_in,
            "charge_out": charge_out,
            "cell_voltages": cellv,
            "temperature_battery": [25.0] * n,
        }
    )


def test_prepare_long_buckets_and_soc_nominal():
    df_long = soh.prepare_long(_synthetic_raw(), CAPACITY_AH, NUM_CELLS)
    assert {"soc_nominal", "current_average", "cell_temperatures", "cell_position"} <= set(
        df_long.columns
    )
    # current_average is floored to a multiple of 5
    avgs = df_long["current_average"].drop_nulls().to_list()
    assert all(abs(a % 5) < 1e-9 for a in avgs)
    # temperature bucketed to 25
    assert df_long["cell_temperatures"].drop_nulls().unique().to_list() == [25.0]


def test_estimate_soh_produces_per_cell_rows(lut):
    df_soh = soh.estimate_soh(
        _synthetic_raw(), lut, CAPACITY_AH, NUM_CELLS, current_threshold=-100.0
    )
    assert not df_soh.is_empty()
    assert "soh_estimate_ffill" in df_soh.columns
    assert "cell_number" in df_soh.columns
    assert sorted(df_soh["cell_number"].unique().to_list()) == list(range(1, NUM_CELLS + 1))


def test_current_threshold_gates_samples(lut):
    raw = _synthetic_raw()
    strict = soh.estimate_soh(raw, lut, CAPACITY_AH, NUM_CELLS, current_threshold=-100.0)
    loose = soh.estimate_soh(raw, lut, CAPACITY_AH, NUM_CELLS, current_threshold=0.0)
    # A more permissive (less negative) gate keeps at least as many samples.
    assert loose.height >= strict.height


def test_summarise_soh_keys(lut):
    df_soh = soh.estimate_soh(
        _synthetic_raw(), lut, CAPACITY_AH, NUM_CELLS, current_threshold=-100.0
    )
    summary = soh.summarise_soh(df_soh, NUM_CELLS)
    for i in range(1, NUM_CELLS + 1):
        assert f"soh_cell_{i}" in summary
    assert "lowest_soh_per_pack" in summary
    assert "time_stamp_of_last_recording" in summary


def test_summarise_soh_empty():
    summary = soh.summarise_soh(pl.DataFrame(), NUM_CELLS)
    assert summary["lowest_soh_per_pack"] is None
    assert all(summary[f"soh_cell_{i}"] is None for i in range(1, NUM_CELLS + 1))
