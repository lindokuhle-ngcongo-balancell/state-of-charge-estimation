"""
soc_reported_vs_estimate.py
───────────────────────────
Fetch the full history of a single battery from Athena, run the per-cell
SOC estimation and display an animated Plotly chart with:
    • soc_nominal            — one line, reported BMS SOC (same for all cells)
    • soc_cell_N             — one line per cell, LUT-based rolling SOC estimate

LUT is built directly from the bundled constant-current CSV files (no external
parquet required).  For any measured current that falls between two LUT bins,
the script:
    1. Finds the closest current bin in the LUT.
    2. Removes the IR voltage drop difference before the lookup:
           V_adjusted = V_measured − R_nominal × (I_actual − I_bin)
       (voltages in mV, R in Ω → multiply R by 1000)
    3. Looks up SOC from the adjusted voltage on that bin's curve.

Temperature compensation is stubbed out (all CSV data is at 25 °C).  The
nearest temperature bin is used; a gradient correction term will be added
later once the equation is known.

Usage
-----
uv run soc_reported_vs_estimate.py \\
    --serial      P34-000092 \\
    --capacity    277 \\
    --num_cells   32 \\
    --csv_dir     data/csv_lut \\          # folder containing the *_Current_*A.csv files
    --ocv_csv     data/ocv_data_for_plotting_q.csv \\  # OCV file; gold-standard for 0 A bin
    --resistance  0.0004 \\                # nominal cell resistance in Ohms (default 0.0004)
    --aws_profile infra_dev \\
    --output_csv  soc_comparison.csv      # optional

The OCV file (--ocv_csv) replaces the constant-current data for the 0 A bin.
It must contain columns: soh, q_removed_ah_ff, cell_voltages.
Only SOH=100 rows are used.  DOD is derived as q_removed_ah_ff / 277 * 100.
"""

import argparse
import os
import re

import boto3
import awswrangler as wr
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE_SOC        = 7
LUT_REFERENCE_CAPACITY = 277.0   # Ah — capacity the CSV curves were measured at
NOMINAL_RESISTANCE     = 0.0004  # Ω  — default cell internal resistance
POSITIVE_LIMIT=5
NEGATIVE_LIMIT=-5

# ── Weights ───────────────────────────────────────────────────────────────────

def make_weights(window_size: int, decay: float = 0.7) -> list[float]:
    x = np.arange(window_size)
    weights = np.exp(decay * (x - (window_size - 1)))
    return (weights / weights.sum()).tolist()


WEIGHTS = make_weights(WINDOW_SIZE_SOC, decay=0.7)


# ── Build LUT from CSV files ──────────────────────────────────────────────────

def _parse_csv_filename(fname: str) -> tuple[str, int] | None:
    """
    Returns (direction, current_A) from filenames like:
        Single_Cell_277Ah_AT_Constant_Charging_Current_30A.csv
        Single_Cell_277Ah_AT_Constant_Discharging_Current_35A.csv
    Returns None if the filename doesn't match.
    """
    m = re.search(r"(Charging|Discharging)_Current_(\d+)A\.csv$", fname, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).capitalize(), int(m.group(2))


def build_lut_from_csvs(
    csv_dir: str,
    capacity_ah: float,
    ocv_csv: str | None = None,
) -> pl.DataFrame:
    """
    Builds the voltage–DOD lookup table from two sources:

    1.  Constant-current CSV files (*Charging_Current_*A.csv /
        *Discharging_Current_*A.csv) found in csv_dir.
        Each file is binned to the nearest 5 A (signed).
        Files that bin to 0 A are *excluded* when an OCV file is provided
        (the OCV curve is the gold-standard for that bin).

    2.  OCV file (ocv_csv, optional) — columns: soh, q_removed_ah_ff,
        cell_voltages.  Only SOH=100 rows are used.
        DOD = round(q_removed_ah_ff / capacity_ah * 100, 0)
        This replaces any constant-current data that would have landed in
        the 0 A bin (2 A and 4 A charging files).

    Returns a Polars DataFrame with columns:
        current_bin        float — signed A, binned to nearest 5
        dod                float — 0–100
        cell_voltages_mean float — mV
    All data is at 25 °C, SOH=100.
    """
    rows = []
    for fname in sorted(os.listdir(csv_dir)):
        parsed = _parse_csv_filename(fname)
        if parsed is None:
            continue
        direction, current_abs = parsed
        sign           = 1 if direction == "Charging" else -1
        current_raw    = sign * current_abs
        current_scaled = current_raw / capacity_ah * LUT_REFERENCE_CAPACITY
        current_bin    = float((current_scaled // 5) * 5)

        # Skip 0 A bin when OCV data is provided — OCV is the gold standard
        if (NEGATIVE_LIMIT<=current_bin<=POSITIVE_LIMIT) and ocv_csv is not None:
            print(f"  Skipping {fname} (0 A bin — superseded by OCV file)")
            continue

        df = pd.read_csv(os.path.join(csv_dir, fname))
        df["soc_int"]  = df["soc"].round(0).astype(int)
        df["volt_mv"]  = (df["voltage"] * 1000).round(0)

        df_g = (
            df.groupby("soc_int")["volt_mv"]
            .mean()
            .round(0)
            .reset_index()
        )
        df_g["dod"]                = (100 - df_g["soc_int"]).astype(float)
        df_g["current_bin"]        = 0 if (NEGATIVE_LIMIT<=current_bin<=POSITIVE_LIMIT) else current_bin
        df_g["cell_voltages_mean"] = df_g["volt_mv"]
        rows.append(df_g[["current_bin", "dod", "cell_voltages_mean"]])

    if not rows and ocv_csv is None:
        raise FileNotFoundError(
            f"No CSV LUT files found in '{csv_dir}'.  "
            "Expected filenames containing 'Charging_Current_<N>A' or "
            "'Discharging_Current_<N>A'."
        )

    # Average any non-OCV bins that happen to share the same current_bin
    combined = pd.concat(rows, ignore_index=True)
    combined = (
        combined
        .groupby(["current_bin", "dod"])["cell_voltages_mean"]
        .mean()
        .round(0)
        .reset_index()
    )

    # ── OCV data: gold-standard for 0 A bin ──────────────────────────────────
    if ocv_csv is not None:
        df_ocv = pd.read_csv(ocv_csv)
        df_ocv = df_ocv[df_ocv["soh"] == 100.0].copy()
        if df_ocv.empty:
            raise ValueError(f"OCV file '{ocv_csv}' contains no rows with soh=100.")
        df_ocv["dod"]     = (df_ocv["q_removed_ah_ff"] / capacity_ah * 100).round(0)
        df_ocv["volt_mv"] = (df_ocv["cell_voltages"] * 1000).round(0)
        df_ocv_g = (
            df_ocv.groupby("dod")["volt_mv"]
            .mean()
            .round(0)
            .reset_index()
            .rename(columns={"volt_mv": "cell_voltages_mean"})
        )
        df_ocv_g["current_bin"] = 0.0
        combined = pd.concat(
            [combined, df_ocv_g[["current_bin", "dod", "cell_voltages_mean"]]],
            ignore_index=True,
        )
        print(f"  OCV file loaded: {len(df_ocv_g)} rows for 0 A bin (SOH=100)")

    lut = pl.from_pandas(combined).sort(["current_bin", "dod"])
    bins = sorted(lut["current_bin"].unique().to_list())
    print(f"LUT ready: {len(lut):,} rows across {len(bins)} current bins: {bins}")
    return lut


# ── SOC estimation with IR compensation and nearest-bin matching ──────────────

def vectorised_soc_estimation_ocv_anchored(
    lut: pl.DataFrame,
    df: pl.DataFrame,
    resistance_ohm: float,
    capacity_ah: float,
    current_negative_limit: float = NEGATIVE_LIMIT,
    current_positive_limit: float = POSITIVE_LIMIT,
) -> pl.DataFrame:
    """
    Estimate SOC using OCV lookup only when current is near OCV, then coulomb count
    between OCV anchor points.

    Sign convention:
        soc_delta > 0 means SOC removed / discharge
        soc_delta < 0 means SOC added / charge

    Therefore:
        SOC[k] = SOC_anchor - accumulated_soc_removed_since_anchor

    Returns df with:
        soc_ocv_min, soc_ocv_max:
            SOC estimated directly from the LUT only at near-OCV rows.
            Non-OCV rows are NaN.

        soc_min, soc_max:
            OCV-anchored and coulomb-counted SOC estimates.
    """

    R_mv = resistance_ohm * 1000.0
    capacity_coulombs = capacity_ah * 3600.0

    # Strongly recommended: ensure deterministic order before diff/cumsum.
    df = df.sort("sequence_number")

    lut_bins = np.sort(lut["current_bin"].unique().to_numpy())
    lut_curr = lut["current_bin"].to_numpy()
    lut_dod = lut["dod"].to_numpy()
    lut_volt = lut["cell_voltages_mean"].to_numpy()

    currents_actual = df["current_actual"].to_numpy()
    voltages = df["cell_voltages"].to_numpy()

    charge_in = df["charge_in"].to_numpy()
    charge_out = df["charge_out"].to_numpy()

    n = len(df)

    # -------------------------------------------------------------------------
    # 1. SOC delta from coulomb counting
    # -------------------------------------------------------------------------
    d_charge_in = np.diff(charge_in, prepend=charge_in[0])
    d_charge_out = np.diff(charge_out, prepend=charge_out[0])

    # Positive means SOC removed. Negative means SOC added.
    soc_delta = ((d_charge_in - d_charge_out) / capacity_coulombs) * 100.0

    # Remove impossible first-row / reset artefacts if needed.
    soc_delta[0] = 0.0

    # Optional protection against counter resets or bad rows.
    bad_delta = ~np.isfinite(soc_delta)
    soc_delta[bad_delta] = 0.0

    # -------------------------------------------------------------------------
    # 2. Identify near-OCV rows
    # -------------------------------------------------------------------------
    near_ocv_mask = (
        (currents_actual >= current_negative_limit)
        & (currents_actual <= current_positive_limit)
    )

    soc_ocv_min_arr = np.full(n, np.nan)
    soc_ocv_max_arr = np.full(n, np.nan)

    # -------------------------------------------------------------------------
    # 3. Map current to nearest LUT bin
    # -------------------------------------------------------------------------
    bin_indices = np.searchsorted(lut_bins, currents_actual)
    bin_indices = np.clip(bin_indices, 0, len(lut_bins) - 1)

    left_idx = np.clip(bin_indices - 1, 0, len(lut_bins) - 1)
    right_idx = bin_indices

    dist_left = np.abs(currents_actual - lut_bins[left_idx])
    dist_right = np.abs(currents_actual - lut_bins[right_idx])

    nearest_bins = np.where(
        dist_left <= dist_right,
        lut_bins[left_idx],
        lut_bins[right_idx],
    )

    # -------------------------------------------------------------------------
    # 4. Only do LUT SOC lookup for near-OCV rows
    # -------------------------------------------------------------------------
    for curr_bin in np.unique(nearest_bins[near_ocv_mask]):
        lut_mask = lut_curr == curr_bin
        if not lut_mask.any():
            continue

        bin_dod = lut_dod[lut_mask]
        bin_volt = lut_volt[lut_mask]

        valid = np.isfinite(bin_volt) & (bin_volt >= 0) & np.isfinite(bin_dod)
        bin_dod = bin_dod[valid]
        bin_volt = bin_volt[valid]

        if len(bin_dod) == 0:
            continue

        sort_idx = np.argsort(bin_dod)
        bin_dod = bin_dod[sort_idx]
        bin_volt = bin_volt[sort_idx]

        v_max = bin_volt[0]    # DOD = 0, fully charged
        v_min = bin_volt[-1]   # DOD = 100, fully discharged

        data_mask = near_ocv_mask & (nearest_bins == curr_bin)

        if not data_mask.any():
            continue

        # Near OCV: use measured voltage as-is.
        # Do not IR-correct these anchor rows.
        v_query = voltages[data_mask]

        # Voltage curve decreases with DOD, so flip for searchsorted.
        flipped_volt = bin_volt[::-1]
        flipped_dod = bin_dod[::-1]

        idx_right = np.searchsorted(flipped_volt, v_query, side="left")
        idx_right = idx_right.clip(0, len(flipped_volt) - 1)
        idx_left = (idx_right - 1).clip(0, len(flipped_volt) - 1)

        dod_at_left = flipped_dod[idx_left]
        dod_at_right = flipped_dod[idx_right]

        dod_min_v = np.minimum(dod_at_left, dod_at_right).clip(0, 100)
        dod_max_v = np.maximum(dod_at_left, dod_at_right).clip(0, 100)

        above = v_query > v_max
        below = v_query < v_min

        dod_min_v = np.where(above, 0, np.where(below, 100, dod_min_v))
        dod_max_v = np.where(above, 0, np.where(below, 100, dod_max_v))

        soc_ocv_min_arr[data_mask] = (100.0 - dod_max_v).clip(0, 100)
        soc_ocv_max_arr[data_mask] = (100.0 - dod_min_v).clip(0, 100)

    # -------------------------------------------------------------------------
    # 5. Coulomb count between OCV anchors
    # -------------------------------------------------------------------------
    cum_soc_removed = np.cumsum(soc_delta)

    soc_min_arr = np.full(n, np.nan)
    soc_max_arr = np.full(n, np.nan)

    valid_anchor = np.isfinite(soc_ocv_min_arr) & np.isfinite(soc_ocv_max_arr)

    current_anchor_min = np.nan
    current_anchor_max = np.nan
    current_anchor_cum_removed = np.nan

    for i in range(n):
        if valid_anchor[i]:
            # Reset anchor whenever a new near-OCV value is available.
            current_anchor_min = soc_ocv_min_arr[i]
            current_anchor_max = soc_ocv_max_arr[i]
            current_anchor_cum_removed = cum_soc_removed[i]

            soc_min_arr[i] = current_anchor_min
            soc_max_arr[i] = current_anchor_max

        elif np.isfinite(current_anchor_min):
            # Coulomb-count from the most recent OCV anchor.
            removed_since_anchor = cum_soc_removed[i] - current_anchor_cum_removed

            soc_min_arr[i] = current_anchor_min - removed_since_anchor
            soc_max_arr[i] = current_anchor_max - removed_since_anchor

    soc_min_arr = np.clip(soc_min_arr, 0, 100)
    soc_max_arr = np.clip(soc_max_arr, 0, 100)

    return df.with_columns([
        pl.Series("soc_delta", soc_delta, dtype=pl.Float64),
        pl.Series("soc_ocv_min", soc_ocv_min_arr, dtype=pl.Float64),
        pl.Series("soc_ocv_max", soc_ocv_max_arr, dtype=pl.Float64),
        pl.Series("soc_min", soc_min_arr, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_arr, dtype=pl.Float64),
    ])

def lookup_soc_from_voltage_for_row(
    lut: pl.DataFrame,
    current_actual: float,
    cell_voltage: float,
    negative_limit: float =NEGATIVE_LIMIT,
    positive_limit: float =POSITIVE_LIMIT,
) -> tuple[float, float]:
    """
    Lookup SOC min/max for one row using the nearest LUT current bin.

    This is only intended to be called when the row is near OCV.
    """

    lut_bins = np.sort(lut["current_bin"].unique().to_numpy())

    # Find nearest LUT current bin
    idx = np.searchsorted(lut_bins, current_actual)
    idx = np.clip(idx, 0, len(lut_bins) - 1)

    left_idx = max(idx - 1, 0)
    right_idx = idx

    left_bin = lut_bins[left_idx]
    right_bin = lut_bins[right_idx]

    if abs(current_actual - left_bin) <= abs(current_actual - right_bin):
        nearest_bin = left_bin
    else:
        nearest_bin = right_bin

    # For true OCV behaviour, you may prefer forcing the LUT bin to 0A
    # instead of using nearest_bin. For now, this follows your previous logic.
    lut_bin = lut.filter(pl.col("current_bin") == nearest_bin)

    if lut_bin.height == 0:
        return np.nan, np.nan

    lut_bin = (
        lut_bin
        .filter(
            pl.col("cell_voltages_mean").is_not_null()
            & pl.col("dod").is_not_null()
            & (pl.col("cell_voltages_mean") >= 0)
        )
        .sort("dod")
    )

    if lut_bin.height == 0:
        return np.nan, np.nan

    bin_dod = lut_bin["dod"].to_numpy()
    bin_volt = lut_bin["cell_voltages_mean"].to_numpy()

    v_max = bin_volt[0]     # DOD = 0, fully charged
    v_min = bin_volt[-1]    # DOD = 100, fully discharged

    # If voltage is outside LUT range, clamp.
    if cell_voltage > v_max:
        dod_min = 0.0
        dod_max = 0.0
    elif cell_voltage < v_min:
        dod_min = 100.0
        dod_max = 100.0
    else:
        # Voltage decreases as DOD increases.
        # Flip arrays so voltage increases for searchsorted.
        flipped_volt = bin_volt[::-1]
        flipped_dod = bin_dod[::-1]

        idx_right = np.searchsorted(flipped_volt, cell_voltage, side="left")
        idx_right = np.clip(idx_right, 0, len(flipped_volt) - 1)

        idx_left = np.clip(idx_right - 1, 0, len(flipped_volt) - 1)

        dod_left = flipped_dod[idx_left]
        dod_right = flipped_dod[idx_right]

        dod_min = min(dod_left, dod_right)
        dod_max = max(dod_left, dod_right)

    soc_min = 100.0 - dod_max
    soc_max = 100.0 - dod_min

    soc_min = float(np.clip(soc_min, 0, 100))
    soc_max = float(np.clip(soc_max, 0, 100))

    return soc_min, soc_max
def iter_rows_soc_estimation_ocv_anchored(
    lut: pl.DataFrame,
    df: pl.DataFrame,
    capacity_ah: float,
    current_negative_limit: float = -1.0,
    current_positive_limit: float = 1.0,
) -> pl.DataFrame:
    """
    Easy-to-read row-by-row version.

    SOC is updated from voltage lookup only near OCV.

    Between OCV points, SOC is propagated using coulomb counting.

    Sign convention:
        soc_delta > 0  means SOC removed, so estimated SOC decreases.
        soc_delta < 0  means SOC added, so estimated SOC increases.
    """

    capacity_coulombs = capacity_ah * 3600.0

    df = df.sort("sequence_number")

    soc_delta_list = []
    soc_ocv_min_list = []
    soc_ocv_max_list = []
    soc_min_list = []
    soc_max_list = []
    anchor_active_list = []

    previous_charge_in = None
    previous_charge_out = None

    anchor_soc_min = None
    anchor_soc_max = None
    soc_removed_since_anchor = 0.0

    for row in df.iter_rows(named=True):
        current_actual = row["current_actual"]
        cell_voltage = row["cell_voltages"]
        charge_in = row["charge_in"]
        charge_out = row["charge_out"]

        # ------------------------------------------------------------
        # 1. Calculate coulomb-counting delta for this row
        # ------------------------------------------------------------
        if previous_charge_in is None and previous_charge_out is None:
            soc_delta = 0.0
        else:
            delta_charge_in = charge_in - previous_charge_in
            delta_charge_out = charge_out - previous_charge_out

            # Positive means SOC removed.
            # Negative means SOC added.
            soc_delta = (
                (delta_charge_in - delta_charge_out)
                / capacity_coulombs
            ) * 100.0

            if not np.isfinite(soc_delta):
                soc_delta = 0.0

        previous_charge_in = charge_in
        previous_charge_out = charge_out

        # ------------------------------------------------------------
        # 2. Check whether this row is near OCV
        # ------------------------------------------------------------
        is_near_ocv = (
            current_actual >= current_negative_limit
            and current_actual <= current_positive_limit
        )

        # ------------------------------------------------------------
        # 3. If near OCV, use voltage lookup and reset the anchor
        # ------------------------------------------------------------
        if is_near_ocv:
            soc_ocv_min, soc_ocv_max = lookup_soc_from_voltage_for_row(
                lut=lut,
                current_actual=current_actual,
                cell_voltage=cell_voltage,
                negative_limit=current_negative_limit,
                positive_limit=current_positive_limit,
            )

            if np.isfinite(soc_ocv_min) and np.isfinite(soc_ocv_max):
                anchor_soc_min = soc_ocv_min
                anchor_soc_max = soc_ocv_max
                # soc_removed_since_anchor = 0.0

                soc_min = anchor_soc_min
                soc_max = anchor_soc_max
                anchor_active = True
            else:
                soc_min = np.nan
                soc_max = np.nan
                anchor_active = False

        # ------------------------------------------------------------
        # 4. If not near OCV, coulomb-count from latest OCV anchor
        # ------------------------------------------------------------
        else:
            soc_ocv_min = np.nan
            soc_ocv_max = np.nan

            # if anchor_soc_min is not None and anchor_soc_max is not None:
                # soc_removed_since_anchor += soc_delta

            soc_min = anchor_soc_min + soc_delta    
            soc_max = anchor_soc_max + soc_delta

            soc_min = float(np.clip(soc_min, 0, 100))
            soc_max = float(np.clip(soc_max, 0, 100))

            anchor_active = True


        soc_delta_list.append(soc_delta)
        soc_ocv_min_list.append(soc_ocv_min)
        soc_ocv_max_list.append(soc_ocv_max)
        soc_min_list.append(soc_min)
        soc_max_list.append(soc_max)
        anchor_active_list.append(anchor_active)

    return df.with_columns([
        pl.Series("soc_delta", soc_delta_list, dtype=pl.Float64),
        # pl.Series("soc_ocv_min", soc_ocv_min_list, dtype=pl.Float64),
        # pl.Series("soc_ocv_max", soc_ocv_max_list, dtype=pl.Float64),
        pl.Series("soc_min", soc_min_list, dtype=pl.Float64),
        pl.Series("soc_max", soc_max_list, dtype=pl.Float64),
        # pl.Series("soc_anchor_active", anchor_active_list, dtype=pl.Boolean),
    ])
# ── Athena fetch ──────────────────────────────────────────────────────────────

def build_query(serial: str, num_cells: int, capacity_ah: float) -> str:
    capacity_coulombs = capacity_ah * 3600.0
    cell_voltage_cols = "\n".join(
        f"    cell_data.cell_voltage[{i}] AS cell_voltage_{i},"
        for i in range(1, num_cells + 1)
    )
    return f"""
SELECT DISTINCT
    device_serial,
    sequence_number,
    cast(time_stamp as timestamp(3)) as time_stamp,
    current,
    charge_in,
    charge_out,
{cell_voltage_cols}
    ( (({capacity_coulombs} - (charge_top - (charge_in - charge_out))) / {capacity_coulombs}) * 100) as soc,
    temperature_battery
FROM analytics.batteries_v3
WHERE device_serial = '{serial}'
  AND cell_quantity = {num_cells}
  AND time_stamp >= timestamp '2026-04-15' and time_stamp <= timestamp '2026-05-'
ORDER BY sequence_number ASC
"""


def fetch_data(serial: str, num_cells: int, capacity_ah: float, aws_profile: str) -> pl.DataFrame:
    boto3.setup_default_session(profile_name=aws_profile)
    sql = build_query(serial, num_cells, capacity_ah)
    print(f"Fetching data for {serial} from Athena …")
    pd_df = wr.athena.read_sql_query(
        sql=sql, database="scratch", workgroup="data_consumers"
    )
    return pl.from_pandas(pd_df)


# ── Prepare base DataFrame ────────────────────────────────────────────────────

def prepare_base(df_raw: pl.DataFrame, capacity_ah: float, num_cells: int) -> pl.DataFrame:
    """
    Computes per-row:
        soc_nominal      — BMS-reported SOC
        current_actual   — signed current in reference-capacity-scaled Amps
                           (NOT binned — kept continuous for IR correction)
        cell_temperatures — battery temp floored to nearest 5 °C
        cell_voltage_N   — per-cell voltage in V (cast to Float64)
    """
    return (
        df_raw
        .sort(["device_serial", "sequence_number"])
        .with_columns(
            soc_nominal=pl.col("soc").cast(pl.Float64),
            current_average=(
                (pl.col("charge_in").diff() - pl.col("charge_out").diff()) / 60.0
            ).over("device_serial", order_by="sequence_number"),
            temperature_battery=pl.col("temperature_battery").cast(pl.Float64),
            **{
                f"cell_voltage_{i}": pl.col(f"cell_voltage_{i}").cast(pl.Float64)
                for i in range(1, num_cells + 1)
            },
        )
        .drop_nulls()
        .drop_nans()
        .with_columns(
            # Scale current to reference capacity so it's comparable to LUT bins
            current_actual=(
                pl.col("current_average") / capacity_ah * LUT_REFERENCE_CAPACITY
            ),
            cell_temperatures=(
                (pl.col("temperature_battery") // 5 * 5).cast(pl.Float64)
            ),
        )
    )


# ── Build wide result: one rolling_soc_mean column per cell ──────────────────

def build_wide_result(
    df_base: pl.DataFrame,
    lut: pl.DataFrame,
    num_cells: int,
    resistance_ohm: float,
    capacity_ah: float
) -> pl.DataFrame:
    """
    Returns a wide DataFrame with columns:
        sequence_number, sequence_number_original, day,
        soc_nominal,
        soc_cell_1 … soc_cell_N
    """
    seq_base = (
        df_base
        .select("sequence_number", "soc_nominal")
        .unique("sequence_number")
        .sort("sequence_number")
    )

    for i in range(1, num_cells + 1):
        cell_col = f"cell_voltage_{i}"

        df_cell = (
            df_base
            .filter(pl.col(cell_col) > 0)
            .rename({cell_col: "cell_voltages"})
            .with_columns(
                # Convert V → mV for the LUT lookup
                cell_voltages=pl.col("cell_voltages") * 1000,
            )
            .select(
                "sequence_number",
                "cell_temperatures",
                "current_actual",   # continuous, unquantised
                "cell_voltages",
                "soc_nominal",
                "charge_in", 
                "charge_out"
            )
        )

        est = iter_rows_soc_estimation_ocv_anchored(lut, df_cell, capacity_ah=capacity_ah, current_negative_limit=NEGATIVE_LIMIT, current_positive_limit=POSITIVE_LIMIT)

        cell_estimate = (
            est
            .drop_nulls()
            .drop_nans()
            .with_columns(
                rolling_soc_mean=(
                    ((pl.col("soc_max") + pl.col("soc_min")) / 2)
                    .rolling_mean(window_size=WINDOW_SIZE_SOC, min_samples=WINDOW_SIZE_SOC - 1)
                ),
            )
            .sort("sequence_number")
            .select(
                "sequence_number",
                pl.col("rolling_soc_mean").alias(f"soc_cell_{i}"),
            )
        )

        seq_base = seq_base.join(cell_estimate, on="sequence_number", how="left")
        print(f"  Cell {i}/{num_cells} done", end="\r", flush=True)

    return (
        seq_base
        .sort("sequence_number")
        # .with_columns(
        #     # day=(pl.col("sequence_number") // (1_440 *60)),
        # )
        .with_columns(
            sequence_number_original=pl.col("sequence_number"),
            sequence_number=(
                pl.col("sequence_number") - pl.col("sequence_number").min()
            ),
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Animated day-by-day chart: reported SOC vs per-cell rolling LUT SOC estimate.\n"
            "LUT is built from constant-current CSV files; current is matched to the nearest\n"
            "bin and the voltage is IR-corrected for the current difference."
        )
    )
    parser.add_argument("--serial",      required=True,              help="Device serial, e.g. P34-000092")
    parser.add_argument("--capacity",    required=True,  type=float, help="Battery capacity in Ah")
    parser.add_argument("--num_cells",   required=True,  type=int,   help="Number of cells")
    parser.add_argument(
        "--csv_dir",
        default=".",
        help=(
            "Directory containing the constant-current CSV files "
            "(default: current working directory)"
        ),
    )
    parser.add_argument(
        "--ocv_csv",
        default=None,
        help=(
            "Path to OCV CSV file (columns: soh, q_removed_ah_ff, cell_voltages). "
            "SOH=100 rows are used as the gold-standard 0 A LUT bin, replacing "
            "any constant-current files that bin to 0 A (e.g. 2 A, 4 A charging)."
        ),
    )
    parser.add_argument(
        "--resistance",
        default=NOMINAL_RESISTANCE,
        type=float,
        help=f"Nominal cell internal resistance in Ohms (default: {NOMINAL_RESISTANCE})",
    )
    parser.add_argument("--aws_profile", default="infra_dev",        help="AWS profile name (default: infra_dev)")
    parser.add_argument("--output_csv",  default=None,               help="Optional path to save wide-format CSV")
    args = parser.parse_args()

    # ── Build LUT from CSVs ───────────────────────────────────────────────────
    print(f"Building LUT from CSV files in '{args.csv_dir}' …")
    lut = build_lut_from_csvs(args.csv_dir, args.capacity, ocv_csv=args.ocv_csv)
    print(f"Using nominal resistance: {args.resistance} Ω  ({args.resistance * 1000:.2f} mΩ)")

    # ── Fetch from Athena ─────────────────────────────────────────────────────
    df_raw = fetch_data(
        serial=args.serial,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        aws_profile=args.aws_profile,
    )
    print(f"Fetched {len(df_raw):,} rows for {args.serial}")

    df_base = prepare_base(df_raw, args.capacity, args.num_cells)

    # ── SOC estimation ────────────────────────────────────────────────────────
    print("Running SOC estimation per cell …")
    df_wide = build_wide_result(df_base, lut, args.num_cells, args.resistance, args.capacity)
    print(f"\nWide result: {len(df_wide):,} rows")

    if args.output_csv:
        df_wide.write_csv(args.output_csv)
        print(f"Saved to {args.output_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    cell_cols  = [f"soc_cell_{i}" for i in range(1, args.num_cells + 1)]
    print(df_wide.schema)
    result_long = df_wide.unpivot(
        on=["soc_nominal"] + cell_cols,
        index=["sequence_number", "sequence_number_original"],
        variable_name="variable",
        value_name="value",
    )

    fig = px.line(
        result_long.to_pandas(),
        x="sequence_number",
        y="value",
        color="variable",
        # animation_frame="day",
        hover_data=["sequence_number_original"],
        labels={
            "value":           "SOC (%)",
            "variable":        "Series",
            "sequence_number": "Sequence (within day)",
        },
        title=(
            f"SOC Reported vs Per-Cell LUT Estimate — {args.serial} "
            f"({args.num_cells} cells · {args.capacity} Ah · "
            f"R={args.resistance * 1000:.2f} mΩ)"
        ),
        color_discrete_map={"soc_nominal": "black"},
    )
    fig.update_traces(
        selector=dict(name="soc_nominal"),
        line=dict(width=2.5),
        opacity=1.0,
    )
    fig.update_traces(
        selector=lambda t: t.name != "soc_nominal",
        opacity=0.7,
        line=dict(width=1),
    )
    fig.update_yaxes(range=[-5, 105])
    fig.show()


if __name__ == "__main__":
    main()
