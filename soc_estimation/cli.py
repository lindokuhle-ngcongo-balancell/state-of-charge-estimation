"""Command-line interfaces for SOC estimation."""

import argparse
import sys
from pathlib import Path

from soc_estimation.data_sources import AthenaDataSource, CSVDataSource, MQTTDataSource
from soc_estimation.engine import SOCEstimationEngine
from soc_estimation.utils import setup_logger
from soc_estimation.utils.config import get_athena_config


def create_csv_parser() -> argparse.ArgumentParser:
    """Create argument parser for CSV input."""
    p = argparse.ArgumentParser(
        description="Estimate SOC from CSV data using OCV lookup table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Input CSV file path")
    p.add_argument(
        "--lut_csv",
        required=True,
        help="OCV LUT CSV path (must have 'soc' and 'voltage' columns)",
    )
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    p.add_argument("--num_cells", required=True, type=int, help="Number of cells")
    p.add_argument(
        "--rest_current_a",
        default=1.0,
        type=float,
        help="Pack current threshold for rest window (A, default 1.0)",
    )
    p.add_argument(
        "--rest_min_ticks",
        default=5,
        type=int,
        help="Consecutive ticks required for rest window (default 5 = 5 min)",
    )
    p.add_argument(
        "--voltage_tolerance_mv",
        default=0.0,
        type=float,
        help="Voltage tolerance gate in mV (0 = disabled, default)",
    )
    p.add_argument(
        "--voltage_delta_min_mv",
        default=0.0,
        type=float,
        help="Voltage delta gate in mV (0 = disabled, default)",
    )
    p.add_argument(
        "--output_csv",
        default="soc_estimate_output.csv",
        help="Output CSV file path (default soc_estimate_output.csv)",
    )
    p.add_argument(
        "--log_file",
        default="soc_estimate.log",
        help="Log file path (default soc_estimate.log)",
    )
    return p


def run_csv_estimation(args: argparse.Namespace) -> None:
    """Run SOC estimation from CSV file."""
    logger = setup_logger("soc_estimation", args.log_file)
    logger.info("------ SOC Estimation (CSV) Start ------")
    logger.info("Args: %s", vars(args))

    # Validate input files
    if not Path(args.csv).exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)
    if not Path(args.lut_csv).exists():
        logger.error("OCV LUT not found: %s", args.lut_csv)
        sys.exit(1)

    # Create data source and engine
    try:
        data_source = CSVDataSource(args.csv, args.num_cells, logger)
        engine = SOCEstimationEngine(
            data_source=data_source,
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            ocv_lut_path=args.lut_csv,
            rest_current_a=args.rest_current_a,
            rest_min_ticks=args.rest_min_ticks,
            voltage_tolerance_mv=args.voltage_tolerance_mv,
            voltage_delta_min_mv=args.voltage_delta_min_mv,
            logger=logger,
        )

        # Process all records
        engine.process_all_records(output_csv=args.output_csv)
        logger.info("------ SOC Estimation (CSV) Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


def create_athena_parser() -> argparse.ArgumentParser:
    """Create argument parser for Athena input."""
    env_cfg = get_athena_config()

    p = argparse.ArgumentParser(
        description=(
            "Estimate SOC from AWS Athena using OCV lookup table.\n\n"
            "Connection parameters can also be set via environment variables or a\n"
            ".env file (see .env.example).  CLI flags always take precedence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--serial", required=True, help="Battery serial number (e.g. P24-000104)")
    p.add_argument(
        "--lut_csv",
        required=True,
        help="OCV LUT CSV path (must have 'soc' and 'voltage' columns)",
    )
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    p.add_argument("--num_cells", required=True, type=int, help="Number of cells")
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
        "--aws_profile",
        default=env_cfg.aws_profile,
        help="AWS named profile [env: AWS_PROFILE]",
    )
    p.add_argument(
        "--workgroup",
        default=env_cfg.workgroup,
        help="Athena workgroup [env: ATHENA_WORKGROUP, default: data_engineering]",
    )
    p.add_argument(
        "--s3_output",
        default=env_cfg.s3_output,
        help="S3 URI for query results — omit to use the workgroup default [env: ATHENA_S3_OUTPUT]",
    )
    p.add_argument(
        "--start_date",
        default=None,
        help="Only fetch rows on or after this date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end_date",
        default=None,
        help="Only fetch rows on or before this date (YYYY-MM-DD, inclusive)",
    )
    p.add_argument(
        "--rest_current_a",
        default=1.0,
        type=float,
        help="Pack current threshold (A, default 1.0)",
    )
    p.add_argument(
        "--rest_min_ticks",
        default=5,
        type=int,
        help="Rest window ticks (default 5 = 5 min)",
    )
    p.add_argument(
        "--output_csv",
        default="soc_estimate_athena.csv",
        help="Output CSV file (default soc_estimate_athena.csv)",
    )
    p.add_argument(
        "--log_file",
        default="soc_estimate_athena.log",
        help="Log file path (default soc_estimate_athena.log)",
    )
    return p


def run_athena_estimation(args: argparse.Namespace) -> None:
    """Run SOC estimation from AWS Athena."""
    logger = setup_logger("soc_estimation_athena", args.log_file)
    logger.info("------ SOC Estimation (Athena) Start ------")
    logger.info("Args: %s", vars(args))

    if not Path(args.lut_csv).exists():
        logger.error("OCV LUT not found: %s", args.lut_csv)
        sys.exit(1)

    try:
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
        engine = SOCEstimationEngine(
            data_source=data_source,
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            ocv_lut_path=args.lut_csv,
            rest_current_a=args.rest_current_a,
            rest_min_ticks=args.rest_min_ticks,
            logger=logger,
        )

        engine.process_all_records(output_csv=args.output_csv)
        logger.info("------ SOC Estimation (Athena) Complete ------")

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


def create_mqtt_parser() -> argparse.ArgumentParser:
    """Create argument parser for MQTT input."""
    p = argparse.ArgumentParser(
        description="Real-time SOC estimation from MQTT messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--broker", required=True, help="MQTT broker hostname")
    p.add_argument("--port", default=1883, type=int, help="MQTT port (default 1883)")
    p.add_argument(
        "--gateway",
        action="append",
        dest="gateways",
        help="Gateway ID (can be specified multiple times, e.g., --gateway gw-000027 --gateway gw-000105)",
    )
    p.add_argument(
        "--bem_ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="BEM IDs to listen to (default: 1 2 3 4)",
    )
    p.add_argument(
        "--lut_csv",
        required=True,
        help="OCV LUT CSV path (must have 'soc' and 'voltage' columns)",
    )
    p.add_argument("--capacity", required=True, type=float, help="Battery capacity (Ah)")
    p.add_argument("--num_cells", required=True, type=int, help="Number of cells")
    p.add_argument("--username", default=None, help="MQTT username (optional)")
    p.add_argument("--password", default=None, help="MQTT password (optional)")
    p.add_argument(
        "--use_tls",
        action="store_true",
        help="Enable TLS for MQTT",
    )
    p.add_argument(
        "--output_dir",
        default="./mqtt_soc_logs",
        help="Directory for per-BEM CSV logs (default ./mqtt_soc_logs)",
    )
    p.add_argument(
        "--log_file",
        default="soc_estimate_mqtt.log",
        help="Log file path (default soc_estimate_mqtt.log)",
    )
    return p


def run_mqtt_estimation(args: argparse.Namespace) -> None:
    """Run real-time SOC estimation from MQTT messages."""

    logger = setup_logger("soc_estimation_mqtt", args.log_file)
    logger.info("------ SOC Estimation (MQTT) Start ------")
    logger.info("Args: %s", vars(args))

    # Validate inputs
    if not Path(args.lut_csv).exists():
        logger.error("OCV LUT not found: %s", args.lut_csv)
        sys.exit(1)

    if not args.gateways:
        logger.error("No gateways specified (use --gateway)")
        sys.exit(1)

    # Create data source
    try:
        data_source = MQTTDataSource(
            broker=args.broker,
            gateway_ids=args.gateways,
            bem_ids=args.bem_ids,
            num_cells=args.num_cells,
            port=args.port,
            use_tls=args.use_tls,
            username=args.username,
            password=args.password,
            logger=logger,
        )

        engine = SOCEstimationEngine(
            data_source=data_source,
            num_cells=args.num_cells,
            capacity_ah=args.capacity,
            ocv_lut_path=args.lut_csv,
            logger=logger,
        )

        logger.info("Starting real-time SOC estimation...")
        # Note: MQTT is async; in production, integrate with event loop
        engine.process_all_records(output_csv=None)

    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


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
    """Entry point for MQTT CLI."""
    parser = create_mqtt_parser()
    args = parser.parse_args()
    run_mqtt_estimation(args)


if __name__ == "__main__":
    main_csv()
