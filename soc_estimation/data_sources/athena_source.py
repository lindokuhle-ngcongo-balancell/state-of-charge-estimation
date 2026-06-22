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
    """Query battery data from AWS Athena.

    The Athena table (analytics.batteries_v3) stores cell voltages as an
    array column accessed via ``cell_data.cell_voltage[{i}]``.  SOC is
    computed on the fly from ``charge_top``, ``charge_in``, and
    ``charge_out`` using the battery's rated capacity.

    Connection parameters fall back to environment variables (see .env.example).
    CLI flags always take precedence over environment variables.
    """

    def __init__(
        self,
        serial: str,
        num_cells: int,
        capacity_ah: float,
        database: str | None = None,
        table: str | None = None,
        aws_profile: str | None = None,
        workgroup: str | None = None,
        s3_output: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialise Athena data source.

        Args:
            serial: Battery serial number (e.g. "P24-000104").
            num_cells: Number of cells in the battery pack.
            capacity_ah: Battery rated capacity in Ah (used to compute SOC).
            database: Athena database. Falls back to ATHENA_DATABASE env var.
            table: Athena table. Falls back to ATHENA_TABLE env var.
            aws_profile: AWS named profile. Falls back to AWS_PROFILE env var.
            workgroup: Athena workgroup. Falls back to ATHENA_WORKGROUP env var
                (default "data_consumers").  The workgroup must have a
                default S3 output configured if s3_output is omitted.
            s3_output: Optional S3 URI for query results.  If omitted the
                workgroup's default output location is used.
            start_date: Optional ISO date "YYYY-MM-DD" — only rows with
                time_stamp >= this date are returned.
            end_date: Optional ISO date "YYYY-MM-DD" — only rows with
                time_stamp <= end of this day are returned.
            logger: Optional logger instance.
        """
        env_cfg = get_athena_config()

        self.serial = serial
        self.num_cells = num_cells
        self.capacity_ah = capacity_ah
        self.database = database or env_cfg.database
        self.table = table or env_cfg.table
        self.aws_profile = aws_profile or env_cfg.aws_profile
        self.workgroup = workgroup or env_cfg.workgroup
        self.s3_output = s3_output or env_cfg.s3_output
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logger or logging.getLogger(__name__)

        if not self.database:
            raise ValueError(
                "Athena database not set. Pass --database or set ATHENA_DATABASE in .env"
            )
        if not self.table:
            raise ValueError("Athena table not set. Pass --table or set ATHENA_TABLE in .env")

    def _boto3_session(self) -> boto3.Session:
        """Return a boto3 session, optionally scoped to a named profile."""
        if self.aws_profile:
            return boto3.Session(profile_name=self.aws_profile)
        return boto3.Session()

    def _build_query(self) -> str:
        """Build the Athena SQL query."""
        capacity_coulombs = self.capacity_ah * 3600.0

        cell_voltage_cols = "\n        ".join(
            f"cell_data.cell_voltage[{i}] AS cell_voltage_{i},"
            for i in range(1, self.num_cells + 1)
        )

        date_filters = ""
        if self.start_date:
            date_filters += f"\n    AND time_stamp >= timestamp '{self.start_date}'"
        if self.end_date:
            date_filters += f"\n    AND time_stamp < timestamp '{self.end_date}' + interval '1' day"

        return f"""
    SELECT DISTINCT
        device_serial,
        sequence_number,
        cast(time_stamp as timestamp(3)) as time_stamp,
        current,
        charge_in,
        charge_out,
        {cell_voltage_cols}
        (
            (({capacity_coulombs} - (charge_top - (charge_in - charge_out)))
            / {capacity_coulombs}) * 100
        ) as soc,
        temperature_battery
    FROM {self.database}.{self.table}
    WHERE device_serial = '{self.serial}'
    AND cell_quantity = {self.num_cells}
    {date_filters}
    ORDER BY sequence_number ASC
    """

    def get_records(self) -> Iterator[BatteryDataRecord]:
        """
        Execute the Athena query and iterate over battery records.

        Yields:
            BatteryDataRecord for each measurement row, sorted by sequence_number.
        """
        sql = self._build_query()
        self.logger.info(
            "Querying Athena  database=%s  table=%s  serial=%s  workgroup=%s",
            self.database,
            self.table,
            self.serial,
            self.workgroup,
        )
        if self.start_date or self.end_date:
            self.logger.info(
                "Date filter: %s -> %s",
                self.start_date or "beginning",
                self.end_date or "now",
            )
        self.logger.debug("SQL:\n%s", sql)

        kwargs: dict = {
            "sql": sql,
            "database": self.database,
            "workgroup": self.workgroup,
            "boto3_session": self._boto3_session(),
        }
        if self.s3_output:
            kwargs["s3_output"] = self.s3_output

        try:
            pandas_df = wr.athena.read_sql_query(**kwargs)
        except Exception:
            self.logger.exception("Athena query failed")
            raise

        df = pl.from_pandas(pandas_df).sort("sequence_number")
        self.logger.info("Retrieved %d rows from Athena", df.height)

        # Per-row delta_charge: net charge change since the previous tick
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

            cell_voltages: dict[int, float] = {}
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
