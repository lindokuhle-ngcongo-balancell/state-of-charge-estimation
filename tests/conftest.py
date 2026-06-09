"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def ocv_lut_path() -> Path:
    return FIXTURES_DIR / "ocv_lut.csv"


@pytest.fixture
def sample_csv_path() -> Path:
    return FIXTURES_DIR / "sample_input.csv"


@pytest.fixture
def ocv_curve(ocv_lut_path: Path) -> list[tuple[float, float]]:
    """Load the test OCV curve as (voltage_mv, soc_pct) tuples."""
    from soc_estimation.utils import load_ocv_lut

    return load_ocv_lut(ocv_lut_path)
