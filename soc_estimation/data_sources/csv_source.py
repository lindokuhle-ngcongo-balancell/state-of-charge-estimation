"""CSV file data source."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from soc_estimation.data_sources.base import BatteryDataRecord, DataSource


class CSVDataSource(DataSource):
    """Read battery data from CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        num_cells: int,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize CSV data source.

        Args:
            csv_path: Path to input CSV file
            num_cells: Number of cells in battery pack
            logger: Optional logger instance

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is missing required columns
        """
        self.csv_path = Path(csv_path)
        self.num_cells = num_cells
        self.logger = logger or logging.getLogger(__name__)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Validate CSV structure
        self._df = pl.read_csv(self.csv_path, try_parse_dates=True)
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Verify CSV has required columns."""
        required = {
            "sequence_number",
            "time_stamp",
            "temperature_battery",
            "current",
            "charge_in",
            "charge_out",
            "soc",
        }
        cell_cols = {f"cell_voltage_{i}" for i in range(1, self.num_cells + 1)}
        required.update(cell_cols)

        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

    def get_records(self) -> Iterator[BatteryDataRecord]:
        """Iterate over CSV records."""
        self.logger.info("Reading %d rows from %s", self._df.height, self.csv_path)

        # Sort by sequence number to ensure chronological order
        df = self._df.sort("sequence_number")

        # Compute delta_charge
        df = df.with_columns(
            delta_charge=(
                (pl.col("charge_in").diff() - pl.col("charge_out").diff()).cast(pl.Float64)
            )
        )

        rows = df.to_dicts()
        total = len(rows)
        log_every = max(1, total // 20)

        for idx, row in enumerate(rows):
            if idx % log_every == 0:
                self.logger.info("Reading row %d / %d", idx, total)

            # Extract cell voltages
            cell_voltages = {}
            for i in range(1, self.num_cells + 1):
                col_name = f"cell_voltage_{i}"
                if col_name in row:
                    v = row[col_name]
                    if v is not None:
                        cell_voltages[i] = float(v)

            dc = row.get("delta_charge")
            yield BatteryDataRecord(
                sequence_number=int(row["sequence_number"]),
                time_stamp=str(row["time_stamp"]),
                temperature_battery=float(row["temperature_battery"]),
                current=float(row["current"]) if row["current"] is not None else None,
                cell_voltages=cell_voltages,
                charge_in=float(row["charge_in"]),
                charge_out=float(row["charge_out"]),
                soc_nominal=float(row["soc"]),
                delta_charge=float(dc) if dc is not None else None,
            )

        self.logger.info("Done reading %d rows", total)
