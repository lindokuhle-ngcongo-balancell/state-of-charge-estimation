"""
SOC Estimation Engine - orchestrates algorithm with data sources.

Coordinates reading battery data from any source, running the OCV algorithm,
and producing output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from soc_estimation.core import CellState, RestWindowState, process_cell_tick
from soc_estimation.data_sources.base import DataSource
from soc_estimation.utils import load_ocv_lut


class SOCEstimationEngine:
    """Main engine for SOC estimation."""

    def __init__(
        self,
        data_source: DataSource,
        num_cells: int,
        capacity_ah: float,
        ocv_lut_path: str | Path,
        rest_current_a: float = 1.0,
        rest_min_ticks: int = 5,
        voltage_tolerance_mv: float = 0.0,
        voltage_delta_min_mv: float = 0.0,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize SOC estimation engine.

        Args:
            data_source: Data source to read battery records from
            num_cells: Number of cells in battery pack
            capacity_ah: Battery pack capacity in Ah
            ocv_lut_path: Path to OCV lookup table CSV
            rest_current_a: Pack current threshold for rest window (A)
            rest_min_ticks: Consecutive ticks required for rest window
            voltage_tolerance_mv: Voltage tolerance gate (0 = disabled)
            voltage_delta_min_mv: Voltage delta gate (0 = disabled)
            logger: Optional logger
        """
        self.data_source = data_source
        self.num_cells = num_cells
        self.capacity_ah = capacity_ah
        self.rest_current_a = rest_current_a
        self.rest_min_ticks = rest_min_ticks
        self.voltage_tolerance_mv = voltage_tolerance_mv
        self.voltage_delta_min_mv = voltage_delta_min_mv
        self.logger = logger or logging.getLogger(__name__)

        # Load OCV LUT
        self.ocv_curve = load_ocv_lut(ocv_lut_path, self.logger)

        # Initialize cell and rest state
        self.cell_states = [CellState() for _ in range(num_cells)]
        self.rest_state = RestWindowState(rest_current_a, rest_min_ticks)

        self.logger.info(
            "SOC Engine initialized: num_cells=%d capacity=%.1f Ah "
            "rest_current=±%.2f A rest_min_ticks=%d",
            num_cells,
            capacity_ah,
            rest_current_a,
            rest_min_ticks,
        )

    def process_all_records(self, output_csv: str | Path | None = None) -> list[dict]:
        """
        Process all records from data source and estimate SOC.

        Args:
            output_csv: Optional path to write results CSV

        Returns:
            List of result dictionaries (one per cell per tick)
        """
        self.logger.info("Processing records from data source...")
        self.logger.info(
            "Rest window: ±%.2f A for %d consecutive ticks",
            self.rest_current_a,
            self.rest_min_ticks,
        )

        results: list[dict] = []
        row_count = 0
        log_every = 500  # fixed interval; will update once total is known

        for record in self.data_source:
            row_count += 1

            # Update pack-level rest window state (once per tick, before cell loop)
            rest_qualified = self.rest_state.update(record.current)

            any_lut_reset_this_tick = False

            for cell_idx in range(1, self.num_cells + 1):
                result = process_cell_tick(
                    cell_idx=cell_idx,
                    cell_voltage_v=record.cell_voltages.get(cell_idx),
                    delta_charge=record.delta_charge,
                    cell_state=self.cell_states[cell_idx - 1],
                    ocv_curve=self.ocv_curve,
                    capacity_ah=self.capacity_ah,
                    rest_qualified=rest_qualified,
                    voltage_tolerance_mv=self.voltage_tolerance_mv,
                    voltage_delta_min_mv=self.voltage_delta_min_mv,
                    logger=self.logger,
                )

                if result.mode == "LUT":
                    any_lut_reset_this_tick = True

                # Stamp result with record metadata
                result.sequence_number = record.sequence_number
                result.time_stamp = record.time_stamp
                result.soc_nominal = record.soc_nominal

                results.append(result.to_dict())

            # Reset rest counter after any LUT fire this tick so the next
            # OCV reset requires a fresh rest window (matches soc_sequential_ocv.py)
            if any_lut_reset_this_tick:
                self.rest_state.reset()

            if row_count % log_every == 0:
                self.logger.info("Processed %d records", row_count)

        self.logger.info("Done processing %d records", row_count)

        # Write results to CSV if requested
        if output_csv:
            self._write_results_csv(results, output_csv)

        # Log summary statistics
        self._log_summary_stats()

        return results

    def _write_results_csv(self, results: list[dict], output_csv: str | Path) -> None:
        """Write results to CSV file."""
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        schema = {
            "sequence_number": pl.Int64,
            "cell_idx": pl.Int64,
            "mode": pl.Utf8,
            "soc_min": pl.Float64,
            "soc_max": pl.Float64,
            "soc_mid": pl.Float64,
            "soc_hybrid": pl.Float64,
            "time_stamp": pl.Utf8,
            "soc_nominal": pl.Float64,
        }

        df = pl.DataFrame(results, schema=schema)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_csv)
        self.logger.info("Results written to %s (%d rows)", output_csv, df.height)

    def _log_summary_stats(self) -> None:
        """Log per-cell summary statistics."""
        self.logger.info("Per-cell summary:")
        for i, state in enumerate(self.cell_states, start=1):
            self.logger.info(
                "  cell %2d: reset_count=%d  final_soc=%s",
                i,
                state.reset_count,
                f"{state.prev_soc:.2f}%" if state.prev_soc is not None else "None",
            )
