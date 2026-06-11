"""Example: Athena-based SOC estimation (parquet LUT, BEM-schema output)."""

from soc_estimation import AthenaDataSource, setup_logger
from soc_estimation.bem_estimator import load_lut, make_lut_arrays, run_batch_estimation

logger = setup_logger("example_athena", "example_athena.log")

SERIAL = "P34-000092"
LUT_PATH = "data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
CAPACITY_AH = 277
NUM_CELLS = 32
AWS_PROFILE = "infra_dev"
DATABASE = "battery_data"
TABLE = "measurements"
OUTPUT_CSV = "results/example_athena_soc.csv"

print(f"Loading parquet LUT from {LUT_PATH}...")
lut = make_lut_arrays(load_lut(LUT_PATH))

print(f"Querying Athena for serial {SERIAL}...")
data_source = AthenaDataSource(
    serial=SERIAL,
    num_cells=NUM_CELLS,
    capacity_ah=CAPACITY_AH,
    database=DATABASE,
    table=TABLE,
    aws_profile=AWS_PROFILE,
    logger=logger,
)

print("Processing records...")
written = run_batch_estimation(
    records=data_source.get_records(),
    num_cells=NUM_CELLS,
    capacity_ah=CAPACITY_AH,
    lut=lut,
    output_csv=OUTPUT_CSV,
    logger=logger,
)

print(f"Done! Wrote {written} ticks to {OUTPUT_CSV}")
