"""
Shared parquet-LUT SoC estimator.

The same algorithm used by ``mqtt_bem_logger`` for live BEM datagrams, exposed as
plain functions so it can also drive the CSV and Athena CLIs.

Algorithm
---------
1. Convert pack current (A) → scaled current ``I' = I_A / capacity_ah * 277``
   and bucket to nearest 5 A (``floor(I'/5)*5``).
2. Bucket cell temperature to nearest 5 °C (``floor(T/5)*5``).
3. Look up nearest LUT (temperature, current) bin.
4. IR-compensate the measured cell voltage:
       V_adj = V_meas_mV − R_mΩ · (I_actual − I_lut_bin)
5. Search the LUT V curve for that bin → (soc_min, soc_max).

Optionally per-cell, the caller can wrap ``estimate_cell_soc`` with a
``RollingSoc`` (13-sample deque, ``soc_rolling`` populated once ≥12 samples).
"""

from __future__ import annotations

import csv
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from soc_estimation.data_sources.base import BatteryDataRecord

# ─── Constants ───────────────────────────────────────────────────────────────
WINDOW_SIZE_SOC = 13
MIN_SAMPLES_ROLLING = WINDOW_SIZE_SOC - 1
LUT_REFERENCE_CAPACITY = 277.0
RESISTANCE_OHM = 0.00025
LUT_SOH_MIN = 98
LUT_SOH_MAX = 102


# ─── LUT loading ─────────────────────────────────────────────────────────────
def load_lut(lut_path: str) -> pl.DataFrame:
    """Load and aggregate the parquet LUT into one curve per (temp, current) bin."""
    return (
        pl.scan_parquet(lut_path)
        .with_columns((pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod"))
        .filter(pl.col("dod").is_between(0, 100, closed="both"))
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg(pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean"))
        .sort(["soh", "cell_temperatures", "current_average", "dod"])
        .filter(pl.col("soh").is_between(LUT_SOH_MIN, LUT_SOH_MAX, closed="both"))
        .group_by(["cell_temperatures", "current_average", "dod"])
        .agg(pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean"))
        .with_columns(
            pl.col("dod").count().over(["cell_temperatures", "current_average"]).alias("dod_count")
        )
        .filter(pl.col("dod_count") >= 90)
        .drop("dod_count")
        .sort(["cell_temperatures", "current_average", "dod"])
        .collect()
    )


@dataclass
class LutArrays:
    """Pre-extracted numpy arrays / per-bin caches for fast per-row lookup."""

    unique_temps: np.ndarray
    available_currents: dict
    bin_curves: dict


def make_lut_arrays(lut: pl.DataFrame) -> LutArrays:
    temp = lut["cell_temperatures"].to_numpy().astype(float)
    curr = lut["current_average"].to_numpy().astype(float)
    volt = lut["cell_voltages_mean"].to_numpy().astype(float)
    dod = lut["dod"].to_numpy().astype(float)

    unique_temps = np.unique(temp)
    available_currents = {float(t): np.unique(curr[temp == t]) for t in unique_temps}

    bin_curves: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for t in unique_temps:
        for c in available_currents[float(t)]:
            mask = (temp == t) & (curr == c)
            bd, bv = dod[mask], volt[mask]
            valid = np.isfinite(bv) & np.isfinite(bd) & (bv >= 0)
            bd, bv = bd[valid], bv[valid]
            if bd.size == 0:
                continue
            idx = np.argsort(bd)
            bin_curves[(float(t), float(c))] = (bd[idx], bv[idx])

    return LutArrays(
        unique_temps=unique_temps,
        available_currents=available_currents,
        bin_curves=bin_curves,
    )


# ─── Per-cell estimator ──────────────────────────────────────────────────────
def estimate_cell_soc(
    cell_voltage_mv: float,
    cell_temp_bucket: float,
    current_bucket_scaled: float,
    lut: LutArrays,
    resistance_ohm: float = RESISTANCE_OHM,
) -> tuple[float, float]:
    """Return (soc_min, soc_max) for a single (V_cell_mV, T_°C, I_A_scaled) sample.

    Inputs are expected pre-bucketed; the function still snaps to the nearest LUT
    bin in case the reading sits outside the LUT grid.
    """
    if cell_temp_bucket is None or np.isnan(cell_temp_bucket):
        matched_temp = float(np.median(lut.unique_temps))
    else:
        matched_temp = float(
            lut.unique_temps[np.argmin(np.abs(lut.unique_temps - cell_temp_bucket))]
        )

    avail = lut.available_currents.get(matched_temp)
    if avail is None or avail.size == 0:
        return (float("nan"), float("nan"))

    if current_bucket_scaled is None or np.isnan(current_bucket_scaled):
        matched_curr = float(avail[np.argmin(np.abs(avail))])
    else:
        matched_curr = float(avail[np.argmin(np.abs(avail - current_bucket_scaled))])

    r_mohm = resistance_ohm * 1000.0
    v_adj = cell_voltage_mv - r_mohm * (current_bucket_scaled - matched_curr)

    curve = lut.bin_curves.get((matched_temp, matched_curr))
    if curve is None:
        return (float("nan"), float("nan"))
    bin_dod, bin_volt = curve

    v_high, v_low = bin_volt[0], bin_volt[-1]

    if v_adj > v_high:
        dod_lo = dod_hi = 0.0
    elif v_adj < v_low:
        dod_lo = dod_hi = 100.0
    else:
        flipped_v = bin_volt[::-1]
        flipped_d = bin_dod[::-1]
        idx_right = int(
            np.clip(
                np.searchsorted(flipped_v, v_adj, side="left"),
                0,
                len(flipped_v) - 1,
            )
        )
        idx_left = int(np.clip(idx_right - 1, 0, len(flipped_v) - 1))
        d1, d2 = float(flipped_d[idx_left]), float(flipped_d[idx_right])
        dod_lo = max(min(d1, d2), 0.0)
        dod_hi = min(max(d1, d2), 100.0)

    soc_min = max(0.0, min(100.0, 100.0 - dod_hi))
    soc_max = max(0.0, min(100.0, 100.0 - dod_lo))
    return soc_min, soc_max


# ─── Bucketing helpers ───────────────────────────────────────────────────────
def bucket_current(
    current_a: float | None,
    capacity_ah: float,
    *,
    step_a: float = 5.0,
    reference_capacity: float = LUT_REFERENCE_CAPACITY,
) -> float:
    """Scale current to LUT reference capacity, then floor to nearest ``step_a``."""
    if current_a is None or (isinstance(current_a, float) and np.isnan(current_a)):
        return float("nan")
    scaled = current_a / capacity_ah * reference_capacity
    return float(np.floor(scaled / step_a) * step_a)


def bucket_temperature(temp_c: float | None, *, step_c: float = 5.0) -> float:
    """Floor temperature to nearest ``step_c``; return NaN if missing."""
    if temp_c is None or (isinstance(temp_c, float) and np.isnan(temp_c)):
        return float("nan")
    return float(np.floor(temp_c / step_c) * step_c)


# ─── Rolling-SoC window ──────────────────────────────────────────────────────
class RollingSoc:
    """Per-cell deque that tracks the (soc_min+soc_max)/2 midpoint and reports a
    rolling mean once at least ``MIN_SAMPLES_ROLLING`` samples have arrived."""

    def __init__(self, window: int = WINDOW_SIZE_SOC):
        self._buf: deque[float] = deque(maxlen=window)
        self._min_samples = max(1, window - 1)

    def update(self, soc_min: float, soc_max: float) -> None:
        if np.isfinite(soc_min) and np.isfinite(soc_max):
            self._buf.append((soc_min + soc_max) / 2.0)

    def mean(self) -> float:
        """Return the rolling mean, or NaN until enough samples have accumulated."""
        if len(self._buf) < self._min_samples:
            return float("nan")
        return float(np.mean(self._buf))


# ─── Batch runner (CSV / Athena) ─────────────────────────────────────────────
def _build_header(num_cells: int) -> list[str]:
    cell_v_cols = [f"cellVoltage_{i}" for i in range(1, num_cells + 1)]
    soc_cols: list[str] = []
    for i in range(1, num_cells + 1):
        soc_cols += [f"soc_min_{i}", f"soc_max_{i}", f"soc_rolling_{i}"]
    return [
        "sequence_number",
        "datestamp",
        "soc",
        "bemCurrentInst",
        "extTemp",
        *cell_v_cols,
        *soc_cols,
    ]


def _format(value: float, *, ndigits: int = 3) -> str:
    """Mirror BemLogger's empty-on-NaN, fixed-precision-on-finite formatting."""
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{ndigits}f}"


def run_batch_estimation(
    records: Iterable[BatteryDataRecord],
    num_cells: int,
    capacity_ah: float,
    lut: LutArrays,
    output_csv: str | Path,
    logger: logging.Logger | None = None,
    log_every: int = 1000,
) -> int:
    """Stream records through the BEM algorithm and write a single BEM-schema CSV.

    Returns the number of rows written. Output schema matches the MQTT BEM logger
    (one row per tick, ``soc_min_i / soc_max_i / soc_rolling_i`` per cell), so MQTT,
    CSV and Athena outputs can be compared side-by-side or unioned.

    Conventions used to normalise CSV/Athena rows into the BEM datagram shape:
    - ``cell_voltage_i`` (V) → ``cellVoltage_i`` (mV, ×1000)
    - ``current`` (A) → ``bemCurrentInst`` (mA, ×1000); algorithm uses raw A
    - ``temperature_battery`` (°C) → ``extTemp`` and used as every cell's temp
    """
    log = logger or logging.getLogger(__name__)
    rolling = [RollingSoc() for _ in range(num_cells)]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_build_header(num_cells))

        for record in records:
            current_a = record.current
            current_bucket = bucket_current(current_a, capacity_ah)
            t_bucket = bucket_temperature(record.temperature_battery)

            cell_v_cols: list[str] = []
            soc_cols: list[str] = []

            for cell_idx in range(1, num_cells + 1):
                v_v = record.cell_voltages.get(cell_idx)
                if v_v is None or v_v <= 0:
                    cell_v_cols.append("")
                    soc_cols += ["", "", ""]
                    continue

                v_mv = float(v_v) * 1000.0
                cell_v_cols.append(f"{v_mv:.0f}")

                if np.isnan(current_bucket):
                    soc_cols += ["", "", ""]
                    continue

                soc_min, soc_max = estimate_cell_soc(
                    cell_voltage_mv=v_mv,
                    cell_temp_bucket=t_bucket,
                    current_bucket_scaled=current_bucket,
                    lut=lut,
                )

                roll = rolling[cell_idx - 1]
                roll.update(soc_min, soc_max)
                soc_cols += [_format(soc_min), _format(soc_max), _format(roll.mean())]

            bem_current_ma = (
                f"{current_a * 1000.0:.0f}"
                if current_a is not None and np.isfinite(current_a)
                else ""
            )
            writer.writerow(
                [
                    str(record.sequence_number),
                    str(record.time_stamp) if record.time_stamp else "",
                    f"{record.soc_nominal:.3f}" if record.soc_nominal is not None else "",
                    bem_current_ma,
                    _format(record.temperature_battery, ndigits=2),
                    *cell_v_cols,
                    *soc_cols,
                ]
            )
            written += 1
            if written % log_every == 0:
                log.info("Processed %d records", written)

    log.info("Wrote %d rows to %s", written, out_path)
    return written
