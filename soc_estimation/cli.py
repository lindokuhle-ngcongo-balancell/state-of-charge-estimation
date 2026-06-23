"""Command-line interfaces for SOC estimation."""

import argparse
import sys
from pathlib import Path

import polars as pl

from soc_estimation.bem_estimator import load_lut, make_lut_arrays, run_batch_estimation
from soc_estimation.data_sources import AthenaDataSource, CSVDataSource
from soc_estimation.mqtt_bem_logger import main as mqtt_bem_main
from soc_estimation.soh_estimator import (
    DEFAULT_CURRENT_THRESHOLD,
    DEFAULT_TOP_OF_CHARGE_MV,
    estimate_soh,
    fetch_soh_rows,
    run_soh_batch,
    summarise_soh,
)
from soc_estimation.utils import setup_logger
from soc_estimation.utils.config import get_athena_config


def create_csv_parser() -> argparse.ArgumentParser:
    """Create argument parser for CSV input."""
    p = argparse.ArgumentParser(
        description=(
            "Estimate SOC from a CSV file using the parquet LUT (same algorithm as\n"
            "soc-estimate-mqtt). Writes a BEM-schema output CSV — one row per tick\n"
            "with soc_min_i/soc_max_i/soc_rolling_i per cell."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Input CSV file path")
    p.add_argument(
        "--lut-path",
        required=True,
        help="Path to parquet LUT (temperature × current × DoD, same as MQTT path)",
    )
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    p.add_argument("--num-cells", required=True, type=int, help="Number of cells")
    p.add_argument(
        "--output-csv",
        default="soc_estimate_output.csv",
        help="Output CSV file path (default soc_estimate_output.csv)",
    )
    p.add_argument(
        "--log-file",
        default="soc_estimate.log",
        help="Log file path (default soc_estimate.log)",
    )
    return p


def run_csv_estimation(args: argparse.Namespace) -> None:
    """Run SOC estimation from CSV file."""
    logger = setup_logger("soc_estimation", args.log_file)
    logger.info("------ SOC Estimation (CSV) Start ------")
    logger.info("Args: %s", vars(args))

    if not Path(args.csv).exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)
    if not Path(args.lut_path).exists():
        logger.error("LUT not found: %s", args.lut_path)
        sys.exit(1)

    try:
        logger.info("Loading LUT from %s …", args.lut_path)
        lut_df = load_lut(args.lut_path)
        logger.info("LUT loaded: %d rows", lut_df.height)
        lut = make_lut_arrays(lut_df)

        data_source = CSVDataSource(args.csv, args.num_cells, logger)
        run_batch_estimation(
            records=data_source.get_records(),
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            lut=lut,
            output_csv=args.output_csv,
            logger=logger,
        )
        logger.info("------ SOC Estimation (CSV) Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


def create_athena_parser() -> argparse.ArgumentParser:
    """Create argument parser for Athena input."""
    env_cfg = get_athena_config()

    p = argparse.ArgumentParser(
        description=(
            "Estimate SOC from AWS Athena using the parquet LUT (same algorithm as\n"
            "soc-estimate-mqtt). Writes a BEM-schema output CSV.\n\n"
            "Connection parameters can also be set via environment variables or a\n"
            ".env file (see .env.example).  CLI flags always take precedence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--serial", required=True, help="Battery serial number (e.g. P24-000104)")
    p.add_argument(
        "--lut-path",
        required=True,
        help="Path to parquet LUT (temperature × current × DoD, same as MQTT path)",
    )
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    p.add_argument("--num-cells", required=True, type=int, help="Number of cells")
    p.add_argument(
        "--database",
        default=env_cfg.database or "analytics",
        help="Athena database [env: ATHENA_DATABASE]",
    )
    p.add_argument(
        "--table",
        default=env_cfg.table or "batteries_v3",
        help="Athena table [env: ATHENA_TABLE]",
    )
    p.add_argument(
        "--aws-profile",
        default=env_cfg.aws_profile,
        help="AWS named profile [env: AWS_PROFILE]",
    )
    p.add_argument(
        "--workgroup",
        default=env_cfg.workgroup,
        help="Athena workgroup [env: ATHENA_WORKGROUP, default: data_consumers]",
    )
    p.add_argument(
        "--s3-output",
        default=env_cfg.s3_output,
        help="S3 URI for query results — omit to use the workgroup default [env: ATHENA_S3_OUTPUT]",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Only fetch rows on or after this date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Only fetch rows on or before this date (YYYY-MM-DD, inclusive)",
    )
    p.add_argument(
        "--output-csv",
        default="soc_estimate_athena.csv",
        help="Output CSV file (default soc_estimate_athena.csv)",
    )
    p.add_argument(
        "--log-file",
        default="soc_estimate_athena.log",
        help="Log file path (default soc_estimate_athena.log)",
    )
    return p


def run_athena_estimation(args: argparse.Namespace) -> None:
    """Run SOC estimation from AWS Athena."""
    logger = setup_logger("soc_estimation_athena", args.log_file)
    logger.info("------ SOC Estimation (Athena) Start ------")
    logger.info("Args: %s", vars(args))

    if not Path(args.lut_path).exists():
        logger.error("LUT not found: %s", args.lut_path)
        sys.exit(1)

    try:
        logger.info("Loading LUT from %s …", args.lut_path)
        lut_df = load_lut(args.lut_path)
        logger.info("LUT loaded: %d rows", lut_df.height)
        lut = make_lut_arrays(lut_df)

        data_source = AthenaDataSource(
            serial=args.serial,
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            database=args.database,
            table=args.table,
            aws_profile=args.aws_profile,
            workgroup=args.workgroup,
            s3_output=args.s3_output,
            start_date=args.start_date,
            end_date=args.end_date,
            logger=logger,
        )
        run_batch_estimation(
            records=data_source.get_records(),
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            lut=lut,
            output_csv=args.output_csv,
            logger=logger,
        )
        logger.info("------ SOC Estimation (Athena) Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


def _add_soh_common_args(p: argparse.ArgumentParser, env_cfg) -> None:
    """Shared Athena/LUT/threshold flags for the SoH CLIs."""
    p.add_argument(
        "--lut-path",
        required=True,
        help="Path to parquet LUT (same table as the SoC CLIs)",
    )
    p.add_argument("--num-cells", required=True, type=int, help="Number of cells")
    p.add_argument(
        "--current-threshold",
        type=float,
        default=DEFAULT_CURRENT_THRESHOLD,
        help=(
            "Discharge gate (A, 277 Ah-scaled). Only samples with "
            "current_average <= this value feed the SoH estimate. More negative = "
            "stricter (heavier discharge only). Default %(default)s."
        ),
    )
    p.add_argument(
        "--top-of-charge-mv",
        type=float,
        default=DEFAULT_TOP_OF_CHARGE_MV,
        help=(
            "Resting cell voltage (mV) at/above which the q_top reference latches "
            "(cell treated as full). Default %(default)s."
        ),
    )
    p.add_argument(
        "--database",
        default=env_cfg.database or "analytics",
        help="Athena database [env: ATHENA_DATABASE]",
    )
    p.add_argument(
        "--table",
        default=env_cfg.table or "batteries_v3",
        help="Athena table [env: ATHENA_TABLE]",
    )
    p.add_argument(
        "--aws-profile",
        default=env_cfg.aws_profile,
        help="AWS named profile [env: AWS_PROFILE]",
    )
    p.add_argument(
        "--workgroup",
        default=env_cfg.workgroup,
        help="Athena workgroup [env: ATHENA_WORKGROUP, default: data_consumers]",
    )
    p.add_argument(
        "--s3-output",
        default=env_cfg.s3_output,
        help="S3 URI for query results — omit to use the workgroup default [env: ATHENA_S3_OUTPUT]",
    )
    p.add_argument(
        "--start-date", default=None, help="Only fetch rows on/after this date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Only fetch rows on/before this date (YYYY-MM-DD, inclusive)",
    )


def create_soh_single_parser() -> argparse.ArgumentParser:
    """Create argument parser for single-battery SoH estimation."""
    env_cfg = get_athena_config()
    p = argparse.ArgumentParser(
        description=(
            "Estimate per-cell State-of-Health (SoH) for one battery from AWS\n"
            "Athena, using the same parquet-LUT SoC estimator as soc-estimate-csv\n"
            "/athena/mqtt. Compares charge-counter SoC against the LUT SoC estimate\n"
            "on discharge samples; writes per-cell SoH over time and prints the\n"
            "final per-cell SoH (worst cell = pack SoH)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--serial", required=True, help="Battery serial number (e.g. BE20000157)")
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    _add_soh_common_args(p, env_cfg)
    p.add_argument(
        "--output-csv",
        default="soh_single.csv",
        help="Per-cell-over-time SoH output CSV (default soh_single.csv)",
    )
    p.add_argument(
        "--log-file", default="soh_single.log", help="Log file path (default soh_single.log)"
    )
    return p


def run_soh_single(args: argparse.Namespace) -> None:
    """Run single-battery SoH estimation."""
    logger = setup_logger("soc_estimation_soh", args.log_file)
    logger.info("------ SoH Estimation (single battery) Start ------")
    logger.info("Args: %s", vars(args))

    if not Path(args.lut_path).exists():
        logger.error("LUT not found: %s", args.lut_path)
        sys.exit(1)

    try:
        logger.info("Loading LUT from %s …", args.lut_path)
        lut = make_lut_arrays(load_lut(args.lut_path))

        df_raw = fetch_soh_rows(
            [args.serial],
            args.num_cells,
            database=args.database,
            table=args.table,
            workgroup=args.workgroup,
            aws_profile=args.aws_profile,
            s3_output=args.s3_output,
            start_date=args.start_date,
            end_date=args.end_date,
            logger=logger,
        )
        if df_raw.is_empty():
            logger.error("No data returned for serial %s", args.serial)
            sys.exit(1)
        logger.info("Fetched %d rows for %s", df_raw.height, args.serial)

        df_soh = estimate_soh(
            df_raw,
            lut,
            capacity_ah=args.capacity,
            num_cells=args.num_cells,
            current_threshold=args.current_threshold,
            top_of_charge_mv=args.top_of_charge_mv,
            logger=logger,
        )
        if not df_soh.is_empty():
            df_soh.write_csv(args.output_csv)
            logger.info("Per-cell SoH written to %s (%d rows)", args.output_csv, df_soh.height)
        else:
            logger.warning("No qualifying discharge samples — no per-cell output written")

        summary = summarise_soh(df_soh, args.num_cells)
        for i in range(1, args.num_cells + 1):
            val = summary[f"soh_cell_{i}"]
            logger.info("  cell %2d SoH: %s", i, f"{val:.1f}%" if val is not None else "n/a")
        worst = summary["lowest_soh_per_pack"]
        logger.info("Pack SoH (worst cell): %s", f"{worst:.1f}%" if worst is not None else "n/a")
        logger.info("------ SoH Estimation (single battery) Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


def create_soh_batch_parser() -> argparse.ArgumentParser:
    """Create argument parser for batch SoH summary."""
    env_cfg = get_athena_config()
    p = argparse.ArgumentParser(
        description=(
            "Batch per-cell SoH summary for many batteries listed in a spreadsheet\n"
            "(CSV or XLSX with columns: device_serial, capacity, cell_quantity).\n"
            "Batteries sharing (capacity, cell_quantity) are grouped and fetched in\n"
            "chunks from Athena, then scored with the shared parquet-LUT estimator.\n"
            "Writes one summary row per battery."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input .csv or .xlsx (columns: device_serial, capacity, cell_quantity)",
    )
    _add_soh_common_args(p, env_cfg)
    p.add_argument(
        "--chunk-size", type=int, default=50, help="Max serials per Athena query (default 50)"
    )
    p.add_argument(
        "--output-csv",
        default="soh_summary.csv",
        help="Output summary CSV (default soh_summary.csv)",
    )
    p.add_argument(
        "--log-file", default="soh_batch.log", help="Log file path (default soh_batch.log)"
    )
    return p


def _read_devices(input_path: str) -> pl.DataFrame:
    """Read the device spreadsheet (CSV native, XLSX via pandas)."""
    if input_path.lower().endswith(".csv"):
        return pl.read_csv(input_path)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a core dep
        raise ImportError("pandas is required to read .xlsx input") from exc
    return pl.from_pandas(pd.read_excel(input_path))


def run_soh_batch_cli(args: argparse.Namespace) -> None:
    """Run batch SoH summary."""
    logger = setup_logger("soc_estimation_soh_batch", args.log_file)
    logger.info("------ SoH Batch Summary Start ------")
    logger.info("Args: %s", vars(args))

    if not Path(args.lut_path).exists():
        logger.error("LUT not found: %s", args.lut_path)
        sys.exit(1)
    if not Path(args.input).exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    try:
        devices_df = _read_devices(args.input)
        logger.info("Loaded %d batteries from %s", devices_df.height, args.input)

        logger.info("Loading LUT from %s …", args.lut_path)
        lut = make_lut_arrays(load_lut(args.lut_path))

        run_soh_batch(
            devices_df,
            lut,
            output_csv=args.output_csv,
            database=args.database,
            table=args.table,
            workgroup=args.workgroup,
            aws_profile=args.aws_profile,
            s3_output=args.s3_output,
            current_threshold=args.current_threshold,
            top_of_charge_mv=args.top_of_charge_mv,
            chunk_size=args.chunk_size,
            start_date=args.start_date,
            end_date=args.end_date,
            logger=logger,
        )
        logger.info("------ SoH Batch Summary Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


# MQTT entry point: see soc_estimation.mqtt_bem_logger


def main_csv():
    """Entry point for CSV CLI."""
    parser = create_csv_parser()
    args = parser.parse_args()
    run_csv_estimation(args)


def main_athena():
    """Entry point for Athena CLI."""
    parser = create_athena_parser()
    args = parser.parse_args()
    run_athena_estimation(args)


def main_mqtt():
    """Entry point for MQTT CLI (parquet-LUT BEM logger, see mqtt_bem_logger)."""
    mqtt_bem_main()


def main_soh():
    """Entry point for single-battery SoH CLI."""
    args = create_soh_single_parser().parse_args()
    run_soh_single(args)


def main_soh_batch():
    """Entry point for batch SoH summary CLI."""
    args = create_soh_batch_parser().parse_args()
    run_soh_batch_cli(args)


if __name__ == "__main__":
    main_csv()
