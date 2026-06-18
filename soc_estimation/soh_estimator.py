"""
Per-cell State-of-Health (SoH) estimation built on the shared BEM SoC estimator.

This brings the two archived analysis scripts — ``soh_single_battery`` and
``soh_batch_summary`` — into the package, but swaps their bespoke
``vectorised_soc_estimation`` for the **same** parquet-LUT estimator the CSV,
Athena and MQTT SoC paths already use (``bem_estimator.estimate_cell_soc``).

Algorithm
---------
1. **soc_nominal (charge counter).** Per cell, derive a reference SoC from the
   BMS charge counters via the *q_top* algorithm: latch ``delta_charge`` while a
   cell rests at high voltage (≥3500 mV, idle, ≥10 ticks into the rest), forward
   fill it, and convert the charge removed since that top into a SoC %.
2. **soc estimate (LUT).** Feed each (cell voltage, bucketed temperature,
   bucketed current) sample through ``bem_estimator.estimate_cell_soc`` to get a
   ``(soc_min, soc_max)`` bracket, then take a short rolling mean of the
   midpoint (``rolling_soc_mean``).
3. **soh.** On discharge samples only — ``current_average <= current_threshold``
   (default −100 A, scaled to the 277 Ah LUT reference) — compute

       soh = 100 − (soc_nominal − rolling_soc_mean)

   then smooth (long rolling mean) and forward fill. The last forward-filled
   value is that cell's reported SoH; the pack SoH is the worst (lowest) cell.

The ``current_average`` used for both the LUT lookup and the discharge gate is
the charge-counter-derived current (Δcharge per minute) scaled to the 277 Ah
reference and floored to the nearest 5 A — identical to the archived scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from soc_estimation.bem_estimator import (
    LUT_REFERENCE_CAPACITY,
    RESISTANCE_OHM,
    LutArrays,
    estimate_cell_soc,
)

# ─── Constants ───────────────────────────────────────────────────────────────
# Long smoothing window for the SoH estimate (≈1 day at ~60 s/tick).
WINDOW_SIZE_SOH = 1441
# Short window for the SoC-estimate midpoint before differencing against soc_nominal.
WINDOW_SIZE_SOC = 7
# Default discharge gate: only samples discharging at ≥100 A (277 Ah-scaled) count.
DEFAULT_CURRENT_THRESHOLD = -100.0


# ─── Per-cell soc_nominal — q_top / charge-counter algorithm ─────────────────
def compute_per_cell_soc(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """Compute per-cell ``soc_nominal`` via the q_top / charge-counter algorithm.

    Works on a multi-serial DataFrame; every window operation is partitioned by
    ``device_serial`` so each battery's counters are computed independently.
    Expects ``cell_voltages`` as a per-row list (one entry per cell, in V) — the
    raw shape returned by the Athena ``cell_data.cell_voltage`` array column.

    Returns a long DataFrame with one row per (device_serial, sequence_number,
    cell_position) carrying ``soc_nominal`` (and the binned/derived helpers).
    """
    capacity_coulombs = capacity_ah * 3600.0

    return (
        df_raw.sort(["device_serial", "sequence_number"])
        # ── charge-counter derived columns (row-level, before explode) ────────
        .with_columns(
            pl.lit(list(range(num_cells))).cast(pl.List(pl.Int8)).alias("cell_position"),
            (pl.col("cell_voltages") * 1000).cast(pl.List(pl.Int32)),
            (pl.col("charge_in") - pl.col("charge_out")).alias("delta_charge"),
            # current_average: coulombs per minute → A equivalent
            ((pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0)
            .over("device_serial")
            .alias("current_average"),
        )
        # ── state: 1=charging, 0=discharging, 2=idle ─────────────────────────
        .with_columns(
            pl.when(pl.col("current_average") > 1)
            .then(1)
            .when(pl.col("current_average") < -1)
            .then(0)
            .otherwise(2)
            .cast(pl.UInt8)
            .alias("state"),
        )
        # ── group: consecutive runs of the same state ─────────────────────────
        .with_columns(
            (pl.col("state") != pl.col("state").shift(1))
            .fill_null(True)
            .cum_sum()
            .over(["device_serial"])
            .cast(pl.UInt32)
            .alias("group"),
        )
        # ── position within each (device, group) run ─────────────────────────
        .with_columns(
            pl.col("sequence_number")
            .cum_count()
            .over(["device_serial", "group"])
            .alias("position_in_group"),
        )
        # ── explode to one row per cell ───────────────────────────────────────
        .explode(["cell_voltages", "cell_position"])
        # ── q_top: latch delta_charge when cell is resting at high voltage ────
        .with_columns(
            pl.when(
                (pl.col("cell_voltages") >= 3500)  # mV
                & (pl.col("state") == 2)  # idle
                & (pl.col("position_in_group") >= 10)
            )
            .then(pl.col("delta_charge"))
            .otherwise(None)
            .cast(pl.Int64)
            .forward_fill()
            .over(["device_serial", "cell_position"])
            .alias("q_top"),
        )
        .drop_nulls(subset=["q_top"])
        # ── q_top_ff: never let q_top drop below delta_charge ─────────────────
        .with_columns(
            pl.max_horizontal("q_top", "delta_charge").alias("q_top_ff"),
        )
        # ── q_removed_ah ──────────────────────────────────────────────────────
        .with_columns(
            ((pl.col("q_top_ff") - pl.col("delta_charge")) / 3600.0)
            .cast(pl.Float32)
            .alias("q_removed_ah"),
        )
        # ── per-cell soc_nominal ──────────────────────────────────────────────
        .with_columns(
            (
                (
                    capacity_coulombs
                    - (pl.col("q_top_ff") - (pl.col("charge_in") - pl.col("charge_out")))
                )
                / capacity_coulombs
                * 100
            ).alias("soc_nominal"),
        )
    )


# ─── Preprocessing — bucket temperature & current to the LUT grid ────────────
def prepare_long(
    df_raw: pl.DataFrame,
    capacity_ah: float,
    num_cells: int,
) -> pl.DataFrame:
    """Compute per-cell soc_nominal then bucket temperature and current.

    ``current_average`` is scaled to the 277 Ah LUT reference and floored to the
    nearest 5 A; ``cell_temperatures`` is the pack temperature floored to 5 °C —
    exactly the grid ``bem_estimator.estimate_cell_soc`` expects.
    """
    df_long = compute_per_cell_soc(df_raw, capacity_ah, num_cells)

    return (
        df_long.with_columns(
            current=pl.col("current").cast(pl.Float64),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            cell_voltages=pl.col("cell_voltages").cast(pl.Float64),
            soc_nominal=pl.col("soc_nominal").cast(pl.Float64),
        )
        .with_columns(
            current_scaled=(pl.col("current_average") / capacity_ah * LUT_REFERENCE_CAPACITY),
            cell_temperatures=((pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)),
        )
        .with_columns(current_average=((pl.col("current_scaled") // 5 * 5).cast(pl.Float64)))
    )


# ─── SoC bracket via the shared BEM estimator ────────────────────────────────
def estimate_soc_brackets(
    df_cell: pl.DataFrame,
    lut: LutArrays,
    resistance_ohm: float = RESISTANCE_OHM,
) -> pl.DataFrame:
    """Add ``soc_min`` / ``soc_max`` columns using ``bem_estimator.estimate_cell_soc``.

    This is the unifying change versus the archived scripts: instead of a
    separate vectorised lookup, every (voltage, temp bucket, current bucket)
    sample goes through the exact estimator used by the CSV/Athena/MQTT paths.
    Results are cached per distinct ``(voltage_mV, temp_bucket, current_bucket)``
    key, so the per-row Python call stays cheap across a full battery history.
    """
    v = df_cell["cell_voltages"].to_numpy().astype(float)  # mV
    t = df_cell["cell_temperatures"].to_numpy().astype(float)
    c = df_cell["current_average"].to_numpy().astype(float)

    cache: dict[tuple[float, float, float], tuple[float, float]] = {}
    soc_min = np.empty(len(v), dtype=float)
    soc_max = np.empty(len(v), dtype=float)

    for i in range(len(v)):
        key = (v[i], t[i], c[i])
        bracket = cache.get(key)
        if bracket is None:
            bracket = estimate_cell_soc(
                cell_voltage_mv=v[i],
                cell_temp_bucket=t[i],
                current_bucket_scaled=c[i],
                lut=lut,
                resistance_ohm=resistance_ohm,
            )
            cache[key] = bracket
        soc_min[i], soc_max[i] = bracket

    return df_cell.with_columns(
        pl.Series("soc_min", soc_min, dtype=pl.Float64),
        pl.Series("soc_max", soc_max, dtype=pl.Float64),
    )


# ─── Per-cell SoH pipeline ───────────────────────────────────────────────────
def _soh_pipeline(est: pl.DataFrame, current_threshold: float) -> pl.DataFrame:
    """Rolling SoC midpoint → discharge gate → SoH estimate, smooth, forward fill."""
    return (
        est.drop_nulls()
        .drop_nans()
        .with_columns(
            rolling_soc_mean=(
                ((pl.col("soc_max") + pl.col("soc_min")) / 2).rolling_mean(
                    window_size=WINDOW_SIZE_SOC, min_samples=WINDOW_SIZE_SOC - 1
                )
            ),
        )
        .filter(pl.col("current_average") <= current_threshold)
        .drop_nulls()
        .drop_nans()
        .sort("sequence_number")
        .with_columns(soh_estimate_raw=100 - (pl.col("soc_nominal") - pl.col("rolling_soc_mean")))
        .with_columns(
            soh_estimate_rolling=pl.col("soh_estimate_raw").rolling_mean(
                window_size=WINDOW_SIZE_SOH, min_samples=1
            ),
        )
        .with_columns(soh_estimate_ffill=pl.col("soh_estimate_rolling").forward_fill())
        .filter(pl.col("soh_estimate_ffill") > 0)
    )


def estimate_soh(
    df_raw: pl.DataFrame,
    lut: LutArrays,
    capacity_ah: float,
    num_cells: int,
    current_threshold: float = DEFAULT_CURRENT_THRESHOLD,
    resistance_ohm: float = RESISTANCE_OHM,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Estimate per-cell SoH over time for one or more batteries.

    Returns a long DataFrame — one row per qualifying (cell, tick) — with
    ``soh_estimate_ffill`` plus ``cell_number`` (1-based) and ``device_serial``.
    Empty if no cell produced a qualifying discharge sample.
    """
    log = logger or logging.getLogger(__name__)
    df_long = prepare_long(df_raw, capacity_ah, num_cells)

    results: list[pl.DataFrame] = []
    for i in range(num_cells):
        df_cell = (
            df_long.filter(pl.col("cell_position") == i)
            .filter(pl.col("cell_voltages") > 0)
            .select(
                "device_serial",
                "sequence_number",
                "cell_temperatures",
                "current_average",
                "cell_voltages",
                "soc_nominal",
                "time_stamp",
            )
        )
        if df_cell.is_empty():
            continue

        est = estimate_soc_brackets(df_cell, lut, resistance_ohm)
        cell_soh = _soh_pipeline(est, current_threshold).with_columns(
            pl.lit(i + 1).alias("cell_number")
        )
        if not cell_soh.is_empty():
            results.append(cell_soh)
        log.debug("cell %d: %d qualifying SoH rows", i + 1, cell_soh.height)

    if not results:
        return pl.DataFrame()
    return pl.concat(results, how="diagonal_relaxed")


def summarise_soh(df_soh: pl.DataFrame, num_cells: int) -> dict:
    """Reduce a per-tick SoH DataFrame to one final SoH per cell + the pack worst.

    Returns a flat dict: ``soh_cell_1 … soh_cell_N``, ``lowest_soh_per_pack``
    (worst cell) and ``time_stamp_of_last_recording``.
    """
    soh_per_cell: dict[int, float | None] = {}
    last_ts = None

    for i in range(1, num_cells + 1):
        cell = (
            df_soh.filter(pl.col("cell_number") == i).sort("sequence_number")
            if "cell_number" in df_soh.columns
            else pl.DataFrame()
        )
        soh_per_cell[i] = cell["soh_estimate_ffill"][-1] if not cell.is_empty() else None

    if not df_soh.is_empty() and "time_stamp" in df_soh.columns:
        last_ts = df_soh.sort("sequence_number")["time_stamp"][-1]

    valid = [v for v in soh_per_cell.values() if v is not None]
    return {
        **{f"soh_cell_{i}": soh_per_cell.get(i) for i in range(1, num_cells + 1)},
        "lowest_soh_per_pack": min(valid) if valid else None,
        "time_stamp_of_last_recording": last_ts,
    }


# ─── Athena fetch ─────────────────────────────────────────────────────────────
def _build_soh_query(
    serials: list[str],
    num_cells: int,
    database: str,
    table: str,
    start_date: str | None,
    end_date: str | None,
) -> str:
    """SQL for the raw charge-counter + cell-voltage-array rows the q_top algo needs."""
    serial_list = ", ".join(f"'{s}'" for s in serials)
    date_filters = ""
    if start_date:
        date_filters += f"\n  AND time_stamp >= timestamp '{start_date}'"
    if end_date:
        date_filters += f"\n  AND time_stamp < timestamp '{end_date}' + interval '1' day"
    return f"""
SELECT DISTINCT
    device_serial,
    sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current,
    charge_in,
    charge_out,
    cell_data.cell_voltage   AS cell_voltages,
    temperature_battery
FROM {database}.{table}
WHERE device_serial IN ({serial_list})
  AND cell_quantity = {num_cells}{date_filters}
ORDER BY device_serial, sequence_number ASC
"""


def fetch_soh_rows(
    serials: list[str],
    num_cells: int,
    *,
    database: str,
    table: str,
    workgroup: str,
    aws_profile: str | None = None,
    s3_output: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Fetch raw SoH input rows for a batch of serials from Athena.

    Imports of boto3/awswrangler are deferred so the rest of the module (and the
    test suite) does not require AWS dependencies just to import it.
    """
    import awswrangler as wr
    import boto3

    log = logger or logging.getLogger(__name__)
    sql = _build_soh_query(serials, num_cells, database, table, start_date, end_date)
    log.debug("Athena SQL:\n%s", sql)

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    kwargs: dict = {
        "sql": sql,
        "database": database,
        "workgroup": workgroup,
        "boto3_session": session,
    }
    if s3_output:
        kwargs["s3_output"] = s3_output

    pd_df = wr.athena.read_sql_query(**kwargs)
    if pd_df.empty:
        return pl.DataFrame()
    return pl.from_pandas(pd_df)


# ─── Batch runner over a serials spreadsheet ─────────────────────────────────
def _empty_row(serial: str, capacity_ah: float, num_cells: int) -> dict:
    return {
        "device_serial": serial,
        "capacity": capacity_ah,
        "cell_quantity": num_cells,
        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
        "lowest_soh_per_pack": None,
        "time_stamp_of_last_recording": None,
    }


def run_soh_batch(
    devices_df: pl.DataFrame,
    lut: LutArrays,
    output_csv: str | Path,
    *,
    database: str,
    table: str,
    workgroup: str,
    aws_profile: str | None = None,
    s3_output: str | None = None,
    current_threshold: float = DEFAULT_CURRENT_THRESHOLD,
    resistance_ohm: float = RESISTANCE_OHM,
    chunk_size: int = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Summarise final per-cell SoH for every battery in ``devices_df``.

    ``devices_df`` needs columns ``device_serial``, ``capacity``,
    ``cell_quantity``. Batteries sharing (capacity, cell_quantity) are grouped
    and fetched ``chunk_size`` serials at a time so only one chunk of raw data is
    in memory at once. Writes a one-row-per-battery summary CSV and returns it.
    """
    log = logger or logging.getLogger(__name__)

    required = {"device_serial", "capacity", "cell_quantity"}
    missing = required - set(devices_df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    devices_df = devices_df.with_columns(
        pl.col("capacity").cast(pl.Float64),
        pl.col("cell_quantity").cast(pl.Int32),
        pl.col("device_serial").cast(pl.Utf8),
    )

    groups = (
        devices_df.select(["capacity", "cell_quantity"])
        .unique()
        .sort(["capacity", "cell_quantity"])
    )

    summary_rows: list[dict] = []
    max_cells_seen = 0

    for group_row in groups.iter_rows(named=True):
        capacity_ah = float(group_row["capacity"])
        num_cells = int(group_row["cell_quantity"])
        max_cells_seen = max(max_cells_seen, num_cells)

        serials = devices_df.filter(
            (pl.col("capacity") == capacity_ah) & (pl.col("cell_quantity") == num_cells)
        )["device_serial"].to_list()
        log.info(
            "Group capacity=%s Ah cells=%d — %d batteries (chunk_size=%d)",
            capacity_ah,
            num_cells,
            len(serials),
            chunk_size,
        )

        for chunk_idx, start in enumerate(range(0, len(serials), chunk_size), start=1):
            chunk = serials[start : start + chunk_size]
            log.info("Chunk %d: fetching %d serials from Athena …", chunk_idx, len(chunk))

            try:
                df_chunk_raw = fetch_soh_rows(
                    chunk,
                    num_cells,
                    database=database,
                    table=table,
                    workgroup=workgroup,
                    aws_profile=aws_profile,
                    s3_output=s3_output,
                    start_date=start_date,
                    end_date=end_date,
                    logger=log,
                )
            except Exception:
                log.exception("Athena fetch failed for chunk %d", chunk_idx)
                summary_rows.extend(_empty_row(s, capacity_ah, num_cells) for s in chunk)
                continue

            if df_chunk_raw.is_empty():
                log.warning("No data returned for chunk %d", chunk_idx)
                summary_rows.extend(_empty_row(s, capacity_ah, num_cells) for s in chunk)
                continue

            log.info("%d rows fetched — estimating SoH …", df_chunk_raw.height)
            fetched = set(df_chunk_raw["device_serial"].unique().to_list())

            for serial in chunk:
                if serial not in fetched:
                    log.warning("[%s] not found in Athena results", serial)
                    summary_rows.append(_empty_row(serial, capacity_ah, num_cells))
                    continue

                df_battery = df_chunk_raw.filter(pl.col("device_serial") == serial)
                try:
                    df_soh = estimate_soh(
                        df_battery,
                        lut,
                        capacity_ah,
                        num_cells,
                        current_threshold=current_threshold,
                        resistance_ohm=resistance_ohm,
                        logger=log,
                    )
                    soh_row = summarise_soh(df_soh, num_cells)
                    worst = soh_row["lowest_soh_per_pack"]
                    log.info(
                        "[%s] rows=%d lowest_soh=%s last_ts=%s",
                        serial,
                        df_battery.height,
                        f"{worst:.1f}%" if worst is not None else "n/a",
                        soh_row["time_stamp_of_last_recording"],
                    )
                except Exception:
                    log.exception("[%s] processing error", serial)
                    soh_row = {
                        **{f"soh_cell_{j}": None for j in range(1, num_cells + 1)},
                        "lowest_soh_per_pack": None,
                        "time_stamp_of_last_recording": None,
                    }

                summary_rows.append(
                    {
                        "device_serial": serial,
                        "capacity": capacity_ah,
                        "cell_quantity": num_cells,
                        **soh_row,
                    }
                )

    # Normalise soh_cell_* width across mixed cell counts.
    for row in summary_rows:
        for j in range(int(row["cell_quantity"]) + 1, max_cells_seen + 1):
            row.setdefault(f"soh_cell_{j}", None)

    out_df = pl.DataFrame(summary_rows)
    cell_cols = [
        f"soh_cell_{j}" for j in range(1, max_cells_seen + 1) if f"soh_cell_{j}" in out_df.columns
    ]
    ordered = (
        ["device_serial", "capacity", "cell_quantity"]
        + cell_cols
        + ["lowest_soh_per_pack", "time_stamp_of_last_recording"]
    )
    out_df = out_df.select(ordered)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    log.info("SoH summary written to %s (%d batteries)", out_path, out_df.height)
    return out_df
