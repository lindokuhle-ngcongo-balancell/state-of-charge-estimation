"""Example: CSV-based SOC estimation."""

from pathlib import Path
from soc_estimation import (
    CSVDataSource,
    SOCEstimationEngine,
    setup_logger,
)

# Setup
logger = setup_logger("example_csv", "example_csv.log")

# Configuration
CSV_FILE = "data/P24-000104.csv"
LUT_FILE = "data/csv_lut/lut_ocv.csv"
CAPACITY_AH = 412
NUM_CELLS = 25
OUTPUT_CSV = "results/example_soc_output.csv"

# Create data source
print(f"Loading CSV from {CSV_FILE}...")
data_source = CSVDataSource(
    csv_path=CSV_FILE,
    num_cells=NUM_CELLS,
    logger=logger,
)

# Create engine
print(f"Initializing SOC engine with {NUM_CELLS} cells, {CAPACITY_AH} Ah capacity...")
engine = SOCEstimationEngine(
    data_source=data_source,
    num_cells=NUM_CELLS,
    capacity_ah=CAPACITY_AH,
    ocv_lut_path=LUT_FILE,
    rest_current_a=2.5,        # ±2.5 A rest threshold
    rest_min_ticks=5,          # 5 min = 5 ticks
    logger=logger,
)

# Process
print(f"Processing records and writing results to {OUTPUT_CSV}...")
results = engine.process_all_records(output_csv=OUTPUT_CSV)

print(f"Done! Processed {len(results)} cell-tick entries.")
print(f"Results saved to {OUTPUT_CSV}")
