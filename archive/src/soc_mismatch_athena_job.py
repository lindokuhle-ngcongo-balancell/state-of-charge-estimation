"""
SOC mismatch detector for battery telemetry in AWS Athena, optimized around Polars LazyFrames.

Data boundary rule:
  - awswrangler returns pandas from Athena and accepts pandas for S3 writes.
  - Everything between those boundaries is Polars / NumPy, with Polars LazyFrames
    used for feature engineering, cell explode, aggregation, and backtest summaries.

Core detector flow:
  1. Read telemetry from Athena, or scan a local file for offline work.
  2. Convert to a Polars LazyFrame immediately.
  3. Derive average current with:
       current_avg = (delta_charge_in - delta_charge_out) / (delta_sample_count * 3000)
  4. Get latest charge_capacity_coulombs per device, convert C -> Ah with / 3600.
  5. Normalize current to C-rate per device capacity.
  6. Keep positive, stable, constant-current charge segments.
  7. Explode cell_voltage array and estimate SOC per cell from a C-rate-normalized LUT.
  8. Aggregate back to one row per pack sample and flag mismatches.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import boto3
import numpy as np
import polars as pl

try:
    import awswrangler as wr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "awswrangler is required for Athena/S3 boundaries. Install with: "
        "uv add awswrangler boto3 polars numpy pyarrow"
    ) from exc

LOG = logging.getLogger("soc_mismatch_detector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class DetectorConfig:
    database: str
    table: str
    athena_workgroup: str
    athena_output_s3: Optional[str]
    aws_profile: Optional[str]
    lookback_hours: int
    start_time: Optional[str]
    end_time: Optional[str]
    lookup_csv: Path
    lut_reference_capacity_ah: float
    default_capacity_ah: Optional[float]
    capacity_csv: Optional[Path]
    min_charge_current_c: float
    max_current_cv: float
    min_cc_samples: int
    min_cc_seconds: int
    soc_spread_threshold_pct: float
    cell_deviation_threshold_pct: float
    min_bad_cells: int
    alert_battery_count: int
    output_s3_path: Optional[str]
    sns_topic_arn: Optional[str]
    read_local_file: bool
    local_file_path: Optional[Path]


# ---------------------------------------------------------------------------
# Lookup kernel. Current axis is normalized to C-rate.
# ---------------------------------------------------------------------------


def find_nearest_bin(sorted_bins: np.ndarray, value: float) -> int:
    n = len(sorted_bins)
    if n == 0:
        raise ValueError("lookup table has no C-rate bins")
    if n == 1:
        return 0

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_bins[mid] < value:
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return 0
    if lo == n:
        return n - 1

    d_lo = value - sorted_bins[lo - 1]
    d_hi = sorted_bins[lo] - value
    return lo - 1 if d_lo <= d_hi else lo


def search_soc_in_voltage_row(voltage_row: np.ndarray, voltage_v: float) -> tuple[int, int]:
    if voltage_v <= voltage_row[0]:
        return 0, 0
    if voltage_v >= voltage_row[100]:
        return 100, 100

    soc_min, soc_max = 0, 100
    while soc_max - soc_min > 1:
        mid = (soc_min + soc_max) // 2
        v_mid = voltage_row[mid]
        if voltage_v < v_mid:
            soc_max = mid
        elif voltage_v > v_mid:
            soc_min = mid
        else:
            return mid, mid

    if voltage_row[soc_min] == voltage_v:
        return soc_min, soc_min
    if voltage_row[soc_max] == voltage_v:
        return soc_max, soc_max
    return soc_min, soc_max


def choose_soc(voltage_row: np.ndarray, voltage_v: float, soc_min: int, soc_max: int) -> int:
    if soc_min == soc_max:
        return soc_min
    d_min = abs(voltage_v - voltage_row[soc_min])
    d_max = abs(voltage_v - voltage_row[soc_max])
    return soc_min if d_min <= d_max else soc_max


def estimate_soc(c_rate_bins: np.ndarray, voltage_lut: np.ndarray, c_rate: float, voltage_v: float) -> int | None:
    if c_rate is None or voltage_v is None:
        return None
    if not math.isfinite(float(c_rate)) or not math.isfinite(float(voltage_v)):
        return None
    bin_idx = find_nearest_bin(c_rate_bins, float(c_rate))
    row = voltage_lut[bin_idx]
    lo, hi = search_soc_in_voltage_row(row, float(voltage_v))
    return choose_soc(row, float(voltage_v), lo, hi)


def fill_lut_missing_soc_values(matrix: np.ndarray) -> np.ndarray:
    """Fill missing SOC columns row-wise by linear interpolation."""
    x = np.arange(matrix.shape[1], dtype=float)
    out = matrix.astype(float, copy=True)
    for row_idx in range(out.shape[0]):
        row = out[row_idx]
        ok = np.isfinite(row)
        if not ok.any():
            raise ValueError(f"lookup row {row_idx} has no usable voltages")
        if not ok.all():
            out[row_idx] = np.interp(x, x[ok], row[ok])
    return out


def load_lookup_as_c_rate(
    lookup_csv: Path,
    lut_reference_capacity_ah: float,
    current_col: str = "current_bin",
    soc_col: str = "state_of_charge_recorded",
    voltage_col: str = "voltage_median",
) -> tuple[np.ndarray, np.ndarray]:
    """Load long-form LUT with Polars and return (c_rate_bins, voltage_lut[bin, soc])."""
    if lut_reference_capacity_ah <= 0:
        raise ValueError("lut_reference_capacity_ah must be > 0")

    lut = (
        pl.scan_csv(lookup_csv)
        .select(
            pl.col(soc_col).cast(pl.Int64),
            pl.col(current_col).cast(pl.Float64),
            pl.col(voltage_col).cast(pl.Float64),
        )
        .drop_nulls([soc_col, current_col, voltage_col])
        .collect()
    )

    if lut.is_empty():
        raise ValueError(f"lookup CSV has no usable rows: {lookup_csv}")

    abs_q95 = lut.select(pl.col(current_col).abs().quantile(0.95)).item()
    current_scale = 1000.0 if abs_q95 and abs_q95 > 200 else 1.0

    lut = lut.with_columns(
        pl.when(pl.col(voltage_col).abs() > 20)
        .then(pl.col(voltage_col) / 1000.0)
        .otherwise(pl.col(voltage_col))
        .alias("voltage_v"),
        (pl.col(current_col) / current_scale / lut_reference_capacity_ah).alias("c_rate_bin"),
    )

    wide = (
        lut.pivot(
            index="c_rate_bin",
            on=soc_col,
            values="voltage_v",
            aggregate_function="mean",
        )
        .sort("c_rate_bin")
    )

    for soc in range(101):
        col = str(soc)
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    ordered_cols = [str(i) for i in range(101)]
    c_rate_bins = wide.get_column("c_rate_bin").to_numpy().astype(float)
    voltage_lut = wide.select(ordered_cols).to_numpy().astype(float)
    voltage_lut = fill_lut_missing_soc_values(voltage_lut)
    return c_rate_bins, voltage_lut


# ---------------------------------------------------------------------------
# Athena / local extraction
# ---------------------------------------------------------------------------


def setup_aws(config: DetectorConfig) -> None:
    if config.aws_profile:
        boto3.setup_default_session(profile_name=config.aws_profile)


def athena_time_filter(config: DetectorConfig) -> str:
    if config.start_time and config.end_time:
        return (
            f"time_stamp >= TIMESTAMP '{config.start_time}' "
            f"AND time_stamp < TIMESTAMP '{config.end_time}'"
        )
    if config.start_time:
        return f"time_stamp >= TIMESTAMP '{config.start_time}'"
    if config.end_time:
        return f"time_stamp < TIMESTAMP '{config.end_time}'"
    return f"time_stamp >= current_timestamp - interval '{config.lookback_hours}' hour"


def athena_query(config: DetectorConfig) -> str:
    """Athena query using the cloud schema names, with latest capacity per battery."""
    time_filter = athena_time_filter(config)
    return f"""
WITH latest_capacity AS (
    SELECT
        device_serial,
        max_by(coulombs.charge_capacity, time_stamp) AS latest_charge_capacity_coulombs
    FROM {config.database}.{config.table}
    WHERE coulombs.charge_capacity IS NOT NULL
      AND coulombs.charge_capacity > 0
    GROUP BY device_serial
), telemetry AS (
    SELECT
        device_serial,
        CAST(time_stamp AS timestamp(3)) AS time_stamp,
        sequence_number,
        cell_quantity,
        cell_data.cell_voltage AS cell_voltage,
        soc,
        charge_in,
        charge_out,
        sample_count,
        current_inst,
        current_min,
        current_max,
        coulombs.charge_capacity AS row_charge_capacity_coulombs
    FROM {config.database}.{config.table}
    WHERE {time_filter}
)
SELECT
    t.device_serial,
    t.time_stamp,
    t.sequence_number,
    t.cell_quantity,
    t.cell_voltage,
    t.soc,
    t.charge_in,
    t.charge_out,
    t.sample_count,
    t.current_inst,
    t.current_min,
    t.current_max,
    COALESCE(t.row_charge_capacity_coulombs, lc.latest_charge_capacity_coulombs) AS charge_capacity_coulombs
FROM telemetry t
LEFT JOIN latest_capacity lc
    ON t.device_serial = lc.device_serial
"""


def read_telemetry_from_athena(config: DetectorConfig) -> pl.LazyFrame:
    setup_aws(config)
    sql = athena_query(config)
    LOG.info("Reading telemetry from Athena")
    kwargs: dict[str, Any] = {
        "sql": sql,
        "database": config.database,
        "workgroup": config.athena_workgroup,
    }
    if config.athena_output_s3:
        kwargs["s3_output"] = config.athena_output_s3
    pdf = wr.athena.read_sql_query(**kwargs)
    LOG.info("Read %d Athena rows", len(pdf))
    return pl.from_pandas(pdf).lazy()


def scan_local_file(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.scan_csv(path, infer_schema_length=10000)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path)
    raise ValueError(f"Unsupported local file type for {path}; use parquet, csv, jsonl/ndjson")


def read_telemetry(config: DetectorConfig) -> pl.LazyFrame:
    if config.read_local_file:
        if not config.local_file_path:
            raise ValueError("--local-file-path is required with --read-local-file")
        LOG.info("Scanning local file: %s", config.local_file_path)
        return scan_local_file(config.local_file_path)
    return read_telemetry_from_athena(config)


# ---------------------------------------------------------------------------
# Polars feature engineering
# ---------------------------------------------------------------------------


def parse_cell_voltage(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.tolist() if x is not None]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value if x is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [float(x) for x in parsed if x is not None]
        except json.JSONDecodeError:
            text = text.strip("[]")
            return [float(x.strip()) for x in text.split(",") if x.strip()]
    raise TypeError(f"Unsupported cell_voltage type: {type(value)!r}")


def standardize_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    names = set(lf.collect_schema().names())
    rename_map = {}
    candidates = {
        "cell_data.cell_voltage": "cell_voltage",
        "coulombs.charge_capacity": "charge_capacity_coulombs",
        "charge_capacity": "charge_capacity_coulombs",
        "current": "current_inst",
    }
    for src, dst in candidates.items():
        if src in names and dst not in names:
            rename_map[src] = dst
    if rename_map:
        lf = lf.rename(rename_map)
    return lf


def numeric_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False)


def add_derived_current_and_capacity(lf: pl.LazyFrame, config: DetectorConfig) -> pl.LazyFrame:
    lf = standardize_schema(lf)
    schema_names = set(lf.collect_schema().names())
    required = {
        "device_serial",
        "time_stamp",
        "sequence_number",
        "cell_voltage",
        "soc",
        "charge_in",
        "charge_out",
        "sample_count",
        "current_min",
        "current_max",
    }
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"telemetry is missing required columns: {missing}")

    if "charge_capacity_coulombs" not in schema_names:
        lf = lf.with_columns(pl.lit(None, dtype=pl.Float64).alias("charge_capacity_coulombs"))

    lf = lf.with_columns(
        pl.col("device_serial").cast(pl.String),
        pl.col("time_stamp").cast(pl.Datetime("us"), strict=False),
        pl.col("sequence_number").cast(pl.Int64, strict=False),
        numeric_expr("charge_in"),
        numeric_expr("charge_out"),
        numeric_expr("sample_count"),
        numeric_expr("current_min"),
        numeric_expr("current_max"),
        numeric_expr("soc"),
        numeric_expr("charge_capacity_coulombs"),
    ).sort(["device_serial", "time_stamp", "sequence_number"])

    latest_capacity = (
        lf.filter((pl.col("charge_capacity_coulombs").is_not_null()) & (pl.col("charge_capacity_coulombs") > 0))
        .group_by("device_serial")
        .agg(
            pl.col("charge_capacity_coulombs")
            .sort_by(["time_stamp", "sequence_number"])
            .last()
            .alias("latest_charge_capacity_coulombs")
        )
    )

    lf = lf.join(latest_capacity, on="device_serial", how="left")

    capacity_expr = pl.col("latest_charge_capacity_coulombs") / 3600.0
    if config.capacity_csv:
        caps = (
            pl.scan_csv(config.capacity_csv)
            .select(
                pl.col("device_serial").cast(pl.String),
                pl.col("capacity_ah").cast(pl.Float64, strict=False).alias("fallback_capacity_ah"),
            )
        )
        lf = lf.join(caps, on="device_serial", how="left")
        capacity_expr = pl.coalesce(capacity_expr, pl.col("fallback_capacity_ah"))
    if config.default_capacity_ah is not None:
        capacity_expr = pl.coalesce(capacity_expr, pl.lit(float(config.default_capacity_ah)))

    lf = lf.with_columns(
        capacity_expr.alias("capacity_ah"),
        pl.col("charge_in").diff().over("device_serial").alias("delta_charge_in"),
        pl.col("charge_out").diff().over("device_serial").alias("delta_charge_out"),
        pl.col("sample_count").diff().over("device_serial").alias("delta_sample_count"),
    )

    lf = lf.with_columns(
        (
            (pl.col("delta_charge_in") - pl.col("delta_charge_out"))
            / ((pl.col("delta_sample_count") / 3000.0))
        ).alias("current_avg"),
        (pl.col("current_max") - pl.col("current_min")).abs().alias("current_span"),
    )

    lf = lf.with_columns(
        (pl.col("current_avg") / pl.col("capacity_ah")).alias("current_avg_c"),
        (pl.col("current_span") / pl.col("capacity_ah")).alias("current_span_c"),
    ).with_columns(
        (pl.col("current_avg_c") >= config.min_charge_current_c).alias("is_positive_charge"),
        (
            (pl.col("current_avg_c") >= config.min_charge_current_c)
            # & (pl.col("current_span_c") <= config.max_current_cv)
        ).alias("is_constant_current"),
    )

    lf = lf.with_columns(
        (
            pl.col("is_constant_current")
            != pl.col("is_constant_current").shift(1).over("device_serial").fill_null(False)
        )
        .cast(pl.Int64)
        .cum_sum()
        .over("device_serial")
        .alias("cc_segment_id")
    )

    segments = lf.group_by(["device_serial", "cc_segment_id"]).agg(
        pl.col("is_constant_current").sum().cast(pl.Int64).alias("segment_samples"),
        pl.col("time_stamp").min().alias("segment_start"),
        pl.col("time_stamp").max().alias("segment_end"),
    ).with_columns(
        (pl.col("segment_end") - pl.col("segment_start")).dt.total_seconds().alias("segment_seconds")
    )

    return lf.join(
        segments.select("device_serial", "cc_segment_id", "segment_samples", "segment_seconds"),
        on=["device_serial", "cc_segment_id"],
        how="left",
    ).with_columns(
        (
            pl.col("is_constant_current")
            & (pl.col("segment_samples") >= config.min_cc_samples)
            & (pl.col("segment_seconds") >= config.min_cc_seconds)
            & pl.col("capacity_ah").is_not_null()
            & pl.col("current_avg_c").is_finite()
        ).alias("is_usable_cc")
    )


def cell_voltage_list_expr(schema: pl.Schema) -> pl.Expr:
    dtype = schema.get("cell_voltage")
    if dtype is not None and str(dtype).startswith("List"):
        return pl.col("cell_voltage").cast(pl.List(pl.Float64), strict=False)
    if dtype == pl.String:
        # Fast path for JSON arrays. If your local CSV uses non-JSON comma strings,
        # the Python parser fallback below can still handle them.
        return pl.col("cell_voltage").map_elements(parse_cell_voltage, return_dtype=pl.List(pl.Float64))
    return pl.col("cell_voltage").map_elements(parse_cell_voltage, return_dtype=pl.List(pl.Float64))


def explode_cells(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    voltage_list = cell_voltage_list_expr(schema)
    return (
        lf.with_columns(voltage_list.alias("cell_voltage_list"))
        .filter(pl.col("cell_voltage_list").list.len() > 0)
        .with_columns(
            pl.int_ranges(0, pl.col("cell_voltage_list").list.len()).alias("cell_index")
        )
        .explode(["cell_index", "cell_voltage_list"])
        .rename({"cell_voltage_list": "cell_voltage_raw"})
        .with_columns(
            pl.col("cell_index").cast(pl.Int32),
            pl.when(pl.col("cell_voltage_raw").cast(pl.Float64).abs() > 20)
            .then(pl.col("cell_voltage_raw").cast(pl.Float64) / 1000.0)
            .otherwise(pl.col("cell_voltage_raw").cast(pl.Float64))
            .alias("cell_voltage_v"),
        )
    )


def estimate_cells(lf: pl.LazyFrame, c_rate_bins: np.ndarray, voltage_lut: np.ndarray) -> pl.LazyFrame:
    def estimate_from_struct(row: dict[str, Any]) -> int | None:
        return estimate_soc(c_rate_bins, voltage_lut, row["current_avg_c"], row["cell_voltage_v"])

    return lf.with_columns(
        pl.struct("current_avg_c", "cell_voltage_v")
        .map_elements(estimate_from_struct, return_dtype=pl.Int16)
        .alias("cell_soc_estimated")
    ).with_columns(
        (pl.col("cell_soc_estimated").cast(pl.Float64) - pl.col("soc")).alias("cell_soc_error_vs_pack")
    )


def summarize_mismatches_lf(cell_estimates: pl.LazyFrame, config: DetectorConfig) -> pl.LazyFrame:
    usable = cell_estimates.filter(pl.col("is_usable_cc"))
    per_sample = usable.group_by(
        ["device_serial", "time_stamp", "sequence_number", "cc_segment_id"]
    ).agg(
        pl.col("soc").mean().alias("pack_soc"),
        pl.col("capacity_ah").first().alias("capacity_ah"),
        pl.col("current_avg").mean().alias("current_avg"),
        pl.col("current_avg_c").mean().alias("current_avg_c"),
        pl.col("cell_index").n_unique().alias("cell_count"),
        pl.col("cell_soc_estimated").min().alias("min_cell_soc"),
        pl.col("cell_soc_estimated").max().alias("max_cell_soc"),
        pl.col("cell_soc_estimated").mean().alias("mean_cell_soc"),
        pl.col("cell_soc_error_vs_pack").abs().max().alias("max_abs_cell_error"),
        (pl.col("cell_soc_error_vs_pack").abs() >= config.cell_deviation_threshold_pct)
        .sum()
        .cast(pl.Int64)
        .alias("bad_cell_count"),
        pl.col("segment_samples").first().alias("segment_samples"),
        pl.col("segment_seconds").first().alias("segment_seconds"),
    )
    return per_sample.with_columns(
        (pl.col("max_cell_soc") - pl.col("min_cell_soc")).alias("cell_soc_spread")
    ).with_columns(
        (
            (
                (pl.col("cell_soc_spread") >= config.soc_spread_threshold_pct)
                | (pl.col("max_abs_cell_error") >= config.cell_deviation_threshold_pct)
            )
            & (pl.col("bad_cell_count") >= config.min_bad_cells)
        ).alias("is_soc_mismatch")
    )


def latest_mismatches(per_sample: pl.DataFrame) -> pl.DataFrame:
    if per_sample.is_empty():
        return per_sample
    return (
        per_sample.filter(pl.col("is_soc_mismatch"))
        .sort(["device_serial", "time_stamp", "sequence_number"])
        .group_by("device_serial", maintain_order=True)
        .tail(1)
        .sort(["cell_soc_spread", "max_abs_cell_error"], descending=[True, True])
    )


def build_backtest_summary(per_sample: pl.DataFrame, config: DetectorConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    if per_sample.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    df = per_sample.with_columns(
        (pl.col("mean_cell_soc") - pl.col("pack_soc")).abs().alias("abs_mean_cell_soc_error")
    )

    fleet = df.select(
        pl.lit(config.start_time).alias("start_time"),
        pl.lit(config.end_time).alias("end_time"),
        pl.len().alias("usable_cc_samples"),
        pl.col("device_serial").n_unique().alias("batteries_with_usable_cc"),
        pl.col("is_soc_mismatch").sum().cast(pl.Int64).alias("mismatched_samples"),
        pl.col("is_soc_mismatch").mean().alias("mismatch_sample_rate"),
        pl.col("device_serial").filter(pl.col("is_soc_mismatch")).n_unique().alias("batteries_ever_mismatched"),
        pl.col("abs_mean_cell_soc_error").median().alias("median_abs_mean_cell_soc_error"),
        pl.col("abs_mean_cell_soc_error").quantile(0.95).alias("p95_abs_mean_cell_soc_error"),
        pl.col("cell_soc_spread").median().alias("median_cell_soc_spread"),
        pl.col("cell_soc_spread").quantile(0.95).alias("p95_cell_soc_spread"),
        pl.col("cell_soc_spread").max().alias("max_cell_soc_spread"),
        pl.col("current_avg_c").median().alias("median_current_c"),
    )

    by_battery = (
        df.group_by("device_serial")
        .agg(
            pl.len().alias("usable_cc_samples"),
            pl.col("time_stamp").min().alias("first_seen"),
            pl.col("time_stamp").max().alias("last_seen"),
            pl.col("capacity_ah").median().alias("capacity_ah"),
            pl.col("pack_soc").median().alias("median_pack_soc"),
            pl.col("current_avg_c").median().alias("median_current_c"),
            pl.col("is_soc_mismatch").sum().cast(pl.Int64).alias("mismatch_samples"),
            pl.col("abs_mean_cell_soc_error").median().alias("median_abs_mean_cell_soc_error"),
            pl.col("abs_mean_cell_soc_error").quantile(0.95).alias("p95_abs_mean_cell_soc_error"),
            pl.col("cell_soc_spread").median().alias("median_cell_soc_spread"),
            pl.col("cell_soc_spread").quantile(0.95).alias("p95_cell_soc_spread"),
            pl.col("cell_soc_spread").max().alias("max_cell_soc_spread"),
            pl.col("max_abs_cell_error").max().alias("max_abs_cell_error"),
            pl.col("bad_cell_count").max().alias("max_bad_cell_count"),
        )
        .with_columns(
            (pl.col("mismatch_samples") / pl.col("usable_cc_samples")).alias("mismatch_sample_rate")
        )
        .sort(["mismatch_samples", "max_cell_soc_spread", "max_abs_cell_error"], descending=[True, True, True])
    )
    return fleet, by_battery


# ---------------------------------------------------------------------------
# Output / notification boundary
# ---------------------------------------------------------------------------


def write_parquet_dataset(df: pl.DataFrame, path: str) -> None:
    # awswrangler boundary: convert Polars -> pandas only here.
    wr.s3.to_parquet(df.to_pandas(), path, dataset=True, mode="overwrite")


def write_outputs(per_sample: pl.DataFrame, latest: pl.DataFrame, config: DetectorConfig) -> None:
    if not config.output_s3_path:
        return
    setup_aws(config)
    base = config.output_s3_path.rstrip("/")
    LOG.info("Writing results to %s", base)
    if not per_sample.is_empty():
        write_parquet_dataset(per_sample, f"{base}/per_sample/")
    if not latest.is_empty():
        write_parquet_dataset(latest, f"{base}/latest_mismatches/")


def write_backtest_outputs(per_sample: pl.DataFrame, config: DetectorConfig) -> None:
    if per_sample.is_empty():
        LOG.warning("No usable constant-current samples found")
        return

    fleet, by_battery = build_backtest_summary(per_sample, config)
    print("\nBacktest fleet summary:")
    print(fleet)

    cols = [
        "device_serial",
        "usable_cc_samples",
        "mismatch_samples",
        "mismatch_sample_rate",
        "median_abs_mean_cell_soc_error",
        "p95_abs_mean_cell_soc_error",
        "p95_cell_soc_spread",
        "max_cell_soc_spread",
        "max_abs_cell_error",
    ]
    print("\nWorst batteries by mismatch/sample spread:")
    print(by_battery.select([c for c in cols if c in by_battery.columns]).head(50))

    if config.output_s3_path:
        setup_aws(config)
        base = config.output_s3_path.rstrip("/")
        write_parquet_dataset(fleet, f"{base}/backtest_fleet_summary/")
        write_parquet_dataset(by_battery, f"{base}/backtest_by_battery/")


def notify(latest: pl.DataFrame, config: DetectorConfig) -> None:
    count = int(latest.height)
    should_alert = count >= config.alert_battery_count
    cols = [
        "device_serial",
        "time_stamp",
        "sequence_number",
        "pack_soc",
        "current_avg_c",
        "cell_count",
        "min_cell_soc",
        "max_cell_soc",
        "cell_soc_spread",
        "max_abs_cell_error",
        "bad_cell_count",
    ]
    batteries = latest.select([c for c in cols if c in latest.columns]).to_dicts() if count else []
    payload = {
        "alert": should_alert,
        "mismatched_battery_count": count,
        "threshold_battery_count": config.alert_battery_count,
        "batteries": batteries,
    }
    message = json.dumps(payload, default=str, indent=2)
    print(message)

    if should_alert and config.sns_topic_arn:
        setup_aws(config)
        boto3.client("sns").publish(
            TopicArn=config.sns_topic_arn,
            Subject=f"SOC mismatch: {count} batteries under constant-current charge",
            Message=message,
        )


def validate_capacity_available(enriched_lf: pl.LazyFrame, config: DetectorConfig) -> None:
    missing = (
        enriched_lf.filter(pl.col("capacity_ah").is_null() | (pl.col("capacity_ah") <= 0))
        .select(pl.col("device_serial").unique().head(20))
        .collect()
    )
    if not missing.is_empty():
        examples = missing.get_column("device_serial").to_list()
        raise ValueError(
            "Missing latest charge_capacity_coulombs for one or more devices; capacity_ah could not be derived. "
            "Use --capacity-csv or --default-capacity-ah only as a temporary fallback. "
            f"Examples: {examples}"
        )


def run(config: DetectorConfig) -> dict[str, int]:
    c_rate_bins, voltage_lut = load_lookup_as_c_rate(
        config.lookup_csv,
        lut_reference_capacity_ah=config.lut_reference_capacity_ah,
    )

    raw_lf = read_telemetry(config)
    enriched_lf = add_derived_current_and_capacity(raw_lf, config)
    validate_capacity_available(enriched_lf, config)

    cells_lf = explode_cells(enriched_lf)
    estimated_lf = estimate_cells(cells_lf, c_rate_bins, voltage_lut)
    per_sample_lf = summarize_mismatches_lf(estimated_lf, config)

    LOG.info("Collecting per-sample detector output")
    per_sample = per_sample_lf.collect()
    latest = latest_mismatches(per_sample)

    write_outputs(per_sample, latest, config)
    write_backtest_outputs(per_sample, config)
    notify(latest, config)
    return {"mismatched_battery_count": int(latest.height)}


def lambda_handler(event, context):  # AWS Lambda entrypoint
    cfg = config_from_env(event or {})
    return run(cfg)


def config_from_env(overrides: dict[str, Any]) -> DetectorConfig:
    def get(name: str, default=None):
        return overrides.get(name, os.getenv(name, default))

    return DetectorConfig(
        database=str(get("ATHENA_DATABASE")),
        table=str(get("ATHENA_TABLE")),
        athena_workgroup=str(get("ATHENA_WORKGROUP", "primary")),
        athena_output_s3=(None if get("ATHENA_OUTPUT_S3") in (None, "") else str(get("ATHENA_OUTPUT_S3"))),
        aws_profile=(None if get("AWS_PROFILE") in (None, "") else str(get("AWS_PROFILE"))),
        lookback_hours=int(get("LOOKBACK_HOURS", 24)),
        start_time=(None if get("START_TIME") in (None, "") else str(get("START_TIME"))),
        end_time=(None if get("END_TIME") in (None, "") else str(get("END_TIME"))),
        lookup_csv=Path(str(get("LOOKUP_CSV", "/opt/data/constant_current_voltage_soc_data.csv"))),
        lut_reference_capacity_ah=float(get("LUT_REFERENCE_CAPACITY_AH", 277.0)),
        default_capacity_ah=(None if get("DEFAULT_CAPACITY_AH") in (None, "") else float(get("DEFAULT_CAPACITY_AH"))),
        capacity_csv=(None if get("CAPACITY_CSV") in (None, "") else Path(str(get("CAPACITY_CSV")))),
        min_charge_current_c=float(get("MIN_CHARGE_CURRENT_C", 0.05)),
        max_current_cv=float(get("MAX_CURRENT_CV", 0.03)),
        min_cc_samples=int(get("MIN_CC_SAMPLES", 5)),
        min_cc_seconds=int(get("MIN_CC_SECONDS", 300)),
        soc_spread_threshold_pct=float(get("SOC_SPREAD_THRESHOLD_PCT", 10.0)),
        cell_deviation_threshold_pct=float(get("CELL_DEVIATION_THRESHOLD_PCT", 10.0)),
        min_bad_cells=int(get("MIN_BAD_CELLS", 1)),
        alert_battery_count=int(get("ALERT_BATTERY_COUNT", 10)),
        output_s3_path=(None if get("OUTPUT_S3_PATH") in (None, "") else str(get("OUTPUT_S3_PATH"))),
        sns_topic_arn=(None if get("SNS_TOPIC_ARN") in (None, "") else str(get("SNS_TOPIC_ARN"))),
        read_local_file=str(get("READ_LOCAL_FILE", "false")).lower() in {"1", "true", "yes", "y"},
        local_file_path=(None if get("LOCAL_FILE_PATH") in (None, "") else Path(str(get("LOCAL_FILE_PATH")))),
    )


def parse_args() -> DetectorConfig:
    p = argparse.ArgumentParser(description="Detect per-cell SOC mismatches under constant-current charging load.")
    p.add_argument("--database", default=os.getenv("ATHENA_DATABASE", ""))
    p.add_argument("--table", default=os.getenv("ATHENA_TABLE", ""))
    p.add_argument("--athena-workgroup", default="primary")
    p.add_argument("--athena-output-s3")
    p.add_argument("--aws-profile", default=os.getenv("AWS_PROFILE"))
    p.add_argument("--lookback-hours", type=int, default=24)
    p.add_argument("--start-time", help="Inclusive Athena timestamp lower bound, e.g. 2026-01-01 00:00:00")
    p.add_argument("--end-time", help="Exclusive Athena timestamp upper bound, e.g. 2026-04-25 00:00:00")
    p.add_argument("--lookup-csv", type=Path, required=True)
    p.add_argument("--lut-reference-capacity-ah", type=float, default=277.0)
    p.add_argument("--default-capacity-ah", type=float)
    p.add_argument("--capacity-csv", type=Path)
    p.add_argument("--min-charge-current-c", type=float, default=0.05)
    p.add_argument("--max-current-cv", type=float, default=0.03)
    p.add_argument("--min-cc-samples", type=int, default=5)
    p.add_argument("--min-cc-seconds", type=int, default=300)
    p.add_argument("--soc-spread-threshold-pct", type=float, default=10.0)
    p.add_argument("--cell-deviation-threshold-pct", type=float, default=10.0)
    p.add_argument("--min-bad-cells", type=int, default=1)
    p.add_argument("--alert-battery-count", type=int, default=10)
    p.add_argument("--output-s3-path")
    p.add_argument("--sns-topic-arn")
    p.add_argument("--read-local-file", action="store_true", help="Scan a local parquet/csv/jsonl file instead of querying Athena.")
    p.add_argument("--local-file-path", type=Path, help="Path to local parquet/csv/jsonl file for --read-local-file.")
    a = p.parse_args()
    return DetectorConfig(**vars(a))


if __name__ == "__main__":
    run(parse_args())
