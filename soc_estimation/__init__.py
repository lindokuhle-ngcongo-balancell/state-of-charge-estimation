"""
State of Charge (SOC) Estimation Package

Provides comprehensive SOC estimation using OCV (Open-Circuit Voltage) lookup
tables with intelligent rest-window gating. Supports multiple input sources:
- CSV files
- AWS Athena queries
- MQTT real-time listeners
"""

from soc_estimation.core import (
    CellState,
    CellTickResult,
    RestWindowState,
    process_cell_tick,
    voltage_to_soc,
)
from soc_estimation.data_sources import (
    AthenaDataSource,
    BatteryDataRecord,
    CSVDataSource,
    DataSource,
)
from soc_estimation.engine import SOCEstimationEngine
from soc_estimation.soh_estimator import (
    DEFAULT_CURRENT_THRESHOLD,
    DEFAULT_TOP_OF_CHARGE_MV,
    estimate_soh,
    run_soh_batch,
    summarise_soh,
)
from soc_estimation.utils import load_ocv_lut, setup_logger

__version__ = "0.1.0"

__all__ = [
    # Core algorithm
    "CellState",
    "CellTickResult",
    "RestWindowState",
    "process_cell_tick",
    "voltage_to_soc",
    # Data sources
    "DataSource",
    "BatteryDataRecord",
    "CSVDataSource",
    "AthenaDataSource",
    # Engine
    "SOCEstimationEngine",
    # SoH estimation
    "estimate_soh",
    "summarise_soh",
    "run_soh_batch",
    "DEFAULT_CURRENT_THRESHOLD",
    "DEFAULT_TOP_OF_CHARGE_MV",
    # Utilities
    "setup_logger",
    "load_ocv_lut",
]
