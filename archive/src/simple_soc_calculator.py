
import argparse
import os
import re

import boto3
import awswrangler as wr
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px

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

