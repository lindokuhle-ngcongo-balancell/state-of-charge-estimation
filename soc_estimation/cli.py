"""Command-line interfaces for SOC estimation."""

import argparse
import sys
from pathlib import Path

from soc_estimation.bem_estimator import load_lut, make_lut_arrays, run_batch_estimation
from soc_estimation.data_sources import AthenaDataSource, CSVDataSource
from soc_estimation.mqtt_bem_logger import main as mqtt_bem_main
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
        help="Athena workgroup [env: ATHENA_WORKGROUP, default: data_engineering]",
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


if __name__ == "__main__":
    main_csv()
