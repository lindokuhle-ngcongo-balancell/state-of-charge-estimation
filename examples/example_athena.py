"""Example: Athena-based SOC estimation."""

from soc_estimation import (
    AthenaDataSource,
    SOCEstimationEngine,
    setup_logger,
)

# Setup
logger = setup_logger("example_athena", "example_athena.log")

# Configuration
SERIAL = "P34-000092"
LUT_FILE = "data/csv_lut/lut_ocv.csv"
CAPACITY_AH = 277
NUM_CELLS = 32
AWS_PROFILE = "infra_dev"
DATABASE = "battery_data"
TABLE = "measurements"
OUTPUT_CSV = "results/example_athena_soc.csv"

# Create data source
print(f"Querying Athena for serial {SERIAL}...")
data_source = AthenaDataSource(
    serial=SERIAL,
    num_cells=NUM_CELLS,
    database=DATABASE,
    table=TABLE,
    aws_profile=AWS_PROFILE,
    logger=logger,
)

# Create engine
print(f"Initializing SOC engine...")
engine = SOCEstimationEngine(
    data_source=data_source,
    num_cells=NUM_CELLS,
    capacity_ah=CAPACITY_AH,
    ocv_lut_path=LUT_FILE,
    logger=logger,
)

# Process
print(f"Processing records...")
results = engine.process_all_records(output_csv=OUTPUT_CSV)

print(f"Done! Processed {len(results)} cell-tick entries.")
print(f"Results saved to {OUTPUT_CSV}")
