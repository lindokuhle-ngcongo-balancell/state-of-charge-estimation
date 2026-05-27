"""AWS Athena data source for querying battery data."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import awswrangler as wr
import boto3
import polars as pl

from soc_estimation.data_sources.base import BatteryDataRecord, DataSource
from soc_estimation.utils.config import get_athena_config


class AthenaDataSource(DataSource):
    """Query battery data from AWS Athena."""

    def __init__(
        self,
        serial: str,
        num_cells: int,
        database: str | None = None,
        table: str | None = None,
        aws_profile: str | None = None,
        s3_output: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize Athena data source.

        Connection parameters default to environment variables when not
        supplied explicitly (see .env.example / AWS_PROFILE, ATHENA_DATABASE,
        ATHENA_TABLE, ATHENA_S3_OUTPUT, AWS_REGION).

        Args:
            serial: Battery serial number (e.g., "P34-000092").
            num_cells: Number of cells in the battery pack.
            database: Athena database name. Falls back to ATHENA_DATABASE env var.
            table: Athena table name. Falls back to ATHENA_TABLE env var.
            aws_profile: AWS named profile. Falls back to AWS_PROFILE env var.
            s3_output: S3 URI for query results. Falls back to ATHENA_S3_OUTPUT env var.
            logger: Optional logger instance.
        """
        env_cfg = get_athena_config()

        self.serial = serial
        self.num_cells = num_cells
        self.database = database or env_cfg.database
        self.table = table or env_cfg.table
        self.aws_profile = aws_profile or env_cfg.aws_profile
        self.s3_output = s3_output or env_cfg.s3_output
        self.logger = logger or logging.getLogger(__name__)

        if not self.database:
            raise ValueError(
                "Athena database not set. Pass --database or set ATHENA_DATABASE in .env"
            )
        if not self.table:
            raise ValueError("Athena table not set. Pass --table or set ATHENA_TABLE in .env")
        if not self.s3_output:
            raise ValueError(
                "Athena S3 output not set. Pass --s3_output or set ATHENA_S3_OUTPUT in .env"
            )

    def _boto3_session(self) -> boto3.Session:
        """Create a boto3 session, optionally using a named profile."""
        if self.aws_profile:
            return boto3.Session(profile_name=self.aws_profile)
        return boto3.Session()

    def get_records(self) -> Iterator[BatteryDataRecord]:
        """
        Query Athena and iterate over battery records sorted by sequence_number.

        Yields:
            BatteryDataRecord for each measurement row.
        """
        self.logger.info(
            "Querying Athena  database=%s  table=%s  serial=%s",
            self.database,
            self.table,
            self.serial,
        )

        cell_cols = ", ".join(f"cell_voltage_{i}" for i in range(1, self.num_cells + 1))
        sql = f"""
        SELECT
            sequence_number, time_stamp, temperature_battery, current,
            charge_in, charge_out, soc,
            {cell_cols}
        FROM {self.database}.{self.table}
        WHERE serial = '{self.serial}'
        ORDER BY sequence_number
        """

        try:
            pandas_df = wr.athena.read_sql_query(
                sql=sql,
                database=self.database,
                s3_output=self.s3_output,
                boto3_session=self._boto3_session(),
            )
        except Exception:
            self.logger.exception("Athena query failed")
            raise

        df = pl.from_pandas(pandas_df).sort("sequence_number")
        self.logger.info("Retrieved %d rows from Athena", df.height)

        # Compute per-row delta_charge (net charge change since previous tick)
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
                self.logger.info("Processing row %d / %d", idx + 1, total)

            cell_voltages = {}
            for i in range(1, self.num_cells + 1):
                v = row.get(f"cell_voltage_{i}")
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
