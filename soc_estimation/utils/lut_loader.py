"""OCV LUT (Open-Circuit Voltage Lookup Table) loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl


def load_ocv_lut(
    lut_csv_path: str | Path,
    logger: logging.Logger | None = None,
) -> list[tuple[float, float]]:
    """
    Load OCV LUT from CSV and return sorted curve.

    The CSV must contain columns: soc (0–100 %) and voltage (mV).
    Returns list of (voltage_mv, soc_pct) tuples sorted ascending by voltage.

    Args:
        lut_csv_path: Path to OCV LUT CSV file
        logger: Optional logger instance

    Returns:
        List of (voltage_mv, soc_pct) tuples sorted by voltage

    Raises:
        ValueError: If CSV is missing required columns
        FileNotFoundError: If CSV file doesn't exist
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    lut_path = Path(lut_csv_path)
    if not lut_path.exists():
        raise FileNotFoundError(f"OCV LUT CSV not found: {lut_path}")

    logger.info("Loading OCV LUT from %s", lut_path)
    df = pl.read_csv(lut_path)

    required = {"soc", "voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OCV LUT CSV missing columns: {sorted(missing)}")

    # Sort ascending by voltage for interpolation
    df = df.sort("voltage")
    curve: list[tuple[float, float]] = [
        (float(r["voltage"]), float(r["soc"])) for r in df.iter_rows(named=True)
    ]

    logger.info(
        "OCV LUT loaded: %d points  voltage range [%.0f, %.0f] mV  SoC range [%.1f, %.1f] %%",
        len(curve),
        curve[0][0],
        curve[-1][0],
        curve[0][1],
        curve[-1][1],
    )
    return curve
