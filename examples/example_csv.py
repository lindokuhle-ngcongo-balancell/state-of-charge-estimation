"""Example: CSV-based SOC estimation (parquet LUT, BEM-schema output)."""

from soc_estimation import CSVDataSource, setup_logger
from soc_estimation.bem_estimator import load_lut, make_lut_arrays, run_batch_estimation

logger = setup_logger("example_csv", "example_csv.log")

CSV_FILE = "data/P24-000104.csv"
LUT_PATH = "data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
CAPACITY_AH = 412
NUM_CELLS = 25
OUTPUT_CSV = "results/example_soc_output.csv"

print(f"Loading parquet LUT from {LUT_PATH}...")
lut = make_lut_arrays(load_lut(LUT_PATH))

print(f"Loading CSV from {CSV_FILE}...")
data_source = CSVDataSource(csv_path=CSV_FILE, num_cells=NUM_CELLS, logger=logger)

print(f"Processing records and writing results to {OUTPUT_CSV}...")
written = run_batch_estimation(
    records=data_source.get_records(),
    num_cells=NUM_CELLS,
    capacity_ah=CAPACITY_AH,
    lut=lut,
    output_csv=OUTPUT_CSV,
    logger=logger,
)

print(f"Done! Wrote {written} ticks to {OUTPUT_CSV}")
