#!/usr/bin/env python3
"""
Battery Metrics Visualization with Dynamic CSV Loading and Interactive Metric Selection.
Automatically discovers and processes all CSVs in a directory, prompts the user to
select which metrics to plot, then renders the chosen subplots with consistent
colours per device_serial for sync analysis.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# LOGGING SETUP
# ============================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"battery_metrics_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {LOG_FILE}")


# ============================================================================
# AVAILABLE METRICS REGISTRY
# Each entry: (column_name, display_title, y_axis_label)
# ============================================================================
ALL_METRICS: list[tuple[str, str, str]] = [
    ("current_average",               "Current Average Over Time",                               "Current (A)"),
    ("cumulative_current_average",    "Cumulative Current Average Over Time",                    "Cumul. Current (A)"),
    ("soc",                           "State of Charge (SOC) Over Time",                         "SOC (%)"),
    ("cumulative_voltage",            "Cumulative Voltage Over Time",                            "Cumul. Voltage (V)"),
    ("cumulative_voltage_average",    "Cumulative Voltage Average Over Time",                    "Cumul. Volt. Avg (V)"),
    ("cumul_charge_current",          "Cumulative Current — Charging (current > 0)",             "Cumul. Charge (A)"),
    ("cumul_discharge_current",       "Cumulative Current — Discharging (current < 0)",          "Cumul. Discharge (A)"),
    ("cumul_charge_energy_vavg",      "Cumulative Charging Energy — V_avg × I (I > 0)",          "Cumul. Energy (V·A)"),
    ("cumul_discharge_energy_vavg",   "Cumulative Discharging Energy — V_avg × I (I < 0)",       "Cumul. Energy (V·A)"),
    ("cumul_charge_energy_vcell",     "Cumulative Charging Energy — V_cell × I (I > 0)",         "Cumul. Energy (V·A)"),
    ("cumul_discharge_energy_vcell",  "Cumulative Discharging Energy — V_cell × I (I < 0)",      "Cumul. Energy (V·A)"),
    ("soc_delta",                     "SOC Delta — abs(reference SOC − device SOC)",             "SOC Delta (%)"),
    ("cumul_resting_current",         "Cumulative Current — Resting band (−1 A < I < 1 A)",      "Cumul. Resting (A)"),
]

# Pre-built default selection (indices into ALL_METRICS, 0-based)
DEFAULT_SELECTION: list[int] = [0, 1, 2, 11]  # current_average, cumul_current_avg, soc, soc_delta


# ============================================================================
# INTERACTIVE METRIC SELECTION
# ============================================================================
def prompt_metric_selection(preselect: list[int] | None = None) -> list[tuple[str, str, str]]:
    """
    Present an interactive numbered menu and return the chosen metric tuples.

    The user may:
      - Type comma-separated indices to toggle individual metrics on/off.
      - Type 'a' to select all.
      - Type 'n' to deselect all.
      - Press Enter (empty input) to confirm the current selection.

    Args:
        preselect: Optional list of 0-based indices to pre-select. Defaults to
                   DEFAULT_SELECTION when None.

    Returns:
        List of (column_name, display_title, y_axis_label) tuples for chosen metrics,
        in registry order.

    Raises:
        SystemExit: If the user confirms with no metrics selected.
    """
    selected: set[int] = set(preselect if preselect is not None else DEFAULT_SELECTION)

    def _render_menu() -> None:
        print("\n" + "─" * 60)
        print("  Select metrics to plot")
        print("  Toggle: enter number(s) separated by commas")
        print("  Commands: a = select all  |  n = deselect all  |  Enter = confirm")
        print("─" * 60)
        for idx, (col, title, _) in enumerate(ALL_METRICS):
            marker = "[x]" if idx in selected else "[ ]"
            print(f"  {marker}  {idx + 1:>2}.  {title}")
        print("─" * 60)
        sel_names = [ALL_METRICS[i][0] for i in sorted(selected)]
        print(f"  Selected ({len(selected)}): {', '.join(sel_names) if sel_names else 'none'}")

    while True:
        _render_menu()
        try:
            raw = input("\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

        if raw == "":
            if not selected:
                print("  No metrics selected — please choose at least one.")
                continue
            break

        if raw == "a":
            selected = set(range(len(ALL_METRICS)))
            continue

        if raw == "n":
            selected = set()
            continue

        # Parse comma-separated numbers
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        valid = True
        for token in tokens:
            if not token.isdigit():
                print(f"  Unrecognised input '{token}' — enter numbers, 'a', 'n', or Enter.")
                valid = False
                break
            idx = int(token) - 1
            if not (0 <= idx < len(ALL_METRICS)):
                print(f"  Number {token} is out of range (1–{len(ALL_METRICS)}).")
                valid = False
                break
        if not valid:
            continue

        for token in tokens:
            idx = int(token) - 1
            if idx in selected:
                selected.discard(idx)
            else:
                selected.add(idx)

    chosen = [ALL_METRICS[i] for i in sorted(selected)]
    logger.info(f"User selected {len(chosen)} metric(s): {[c[0] for c in chosen]}")
    return chosen


# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_process_csv(csv_path: Path) -> pl.DataFrame | None:
    """
    Load a single CSV file, sort by sequence_number, and compute base metrics.

    Derived columns added:
      - current_average : (Δcharge_in − Δcharge_out) / 60
      - total_cell_voltage : sum of cell_voltage_1 … cell_voltage_32

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Polars DataFrame with selected columns, or None on failure.
    """
    logger.info(f"Loading CSV: {csv_path}")
    try:
        df = pl.read_csv(csv_path).sort("sequence_number")

        df = df.with_columns(
            current_average=(
                (pl.col("charge_in").diff().fill_null(0) - pl.col("charge_out").diff().fill_null(0))
                / 60
            )
        )

        cell_voltage_cols = [f"cell_voltage_{i}" for i in range(1, 33)]
        df = df.with_columns(
            total_cell_voltage=sum(pl.col(c) for c in cell_voltage_cols)
        )

        df = df.select(
            "device_serial", "time_stamp", "current_average",
            "soc", "total_cell_voltage", "voltage_average",
        )

        logger.info(f"  Processed: {len(df)} rows, device(s): {df['device_serial'].unique().to_list()}")
        return df

    except Exception as e:
        logger.error(f"  Failed to process {csv_path}: {e}")
        return None


def load_all_csvs_from_directory(data_dir: str | Path) -> list[pl.DataFrame]:
    """
    Discover and load all CSV files from a directory.

    Args:
        data_dir: Path to directory containing CSV files.

    Returns:
        List of loaded DataFrames (failed files are skipped).

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV file(s) in {data_dir}")

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return []

    dataframes = [df for f in csv_files if (df := load_and_process_csv(f)) is not None]
    logger.info(f"Successfully loaded {len(dataframes)}/{len(csv_files)} file(s)")
    return dataframes


# ============================================================================
# CUMULATIVE HELPERS
# ============================================================================
def _cumsum_masked(col: str, mask_expr: pl.Expr) -> pl.Expr:
    """
    Cumulative sum of *col* where *mask_expr* is True; zero-fills excluded rows
    so the line stays flat rather than creating time-gaps.

    Args:
        col:       Column name to accumulate.
        mask_expr: Boolean Polars expression selecting contributing rows.

    Returns:
        Polars expression for the masked cumulative sum.
    """
    return pl.when(mask_expr).then(pl.col(col)).otherwise(0.0).cum_sum()


def _cumsum_energy_masked(current_col: str, voltage_col: str, mask_expr: pl.Expr) -> pl.Expr:
    """
    Cumulative sum of voltage × current where *mask_expr* is True; zero-fills
    excluded rows.

    Args:
        current_col: Column name for current values.
        voltage_col: Column name for voltage values.
        mask_expr:   Boolean Polars expression selecting contributing rows.

    Returns:
        Polars expression for the masked cumulative V×A product sum.
    """
    return (
        pl.when(mask_expr)
        .then(pl.col(voltage_col) * pl.col(current_col))
        .otherwise(0.0)
        .cum_sum()
    )


# ============================================================================
# METRIC DERIVATION
# ============================================================================
def derive_all_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute every possible derived metric column in one pass.
    Only columns required for the user's selection will actually be plotted,
    but computing all up-front keeps the derivation logic in one place.

    Args:
        df: Merged DataFrame with base columns loaded from CSVs.

    Returns:
        DataFrame extended with all derived metric columns and soc_delta.
    """
    logger.info("Deriving all cumulative metrics per device")

    charging    = pl.col("current_average") > 5
    discharging = pl.col("current_average") < -5
    resting     = (pl.col("current_average") > -1.0) & (pl.col("current_average") < 0.0)

    df = df.with_columns(
        cumulative_current_average=(
            (pl.col("current_average") / 60)
            .cum_sum()
            .over("device_serial", order_by="time_stamp")
        ),
        cumulative_voltage=(
            pl.col("total_cell_voltage")
            .cum_sum()
            .over("device_serial", order_by="time_stamp")
        ),
        cumulative_voltage_average=(
            pl.col("voltage_average")
            .cum_sum()
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_charge_current=(
            _cumsum_masked("current_average", charging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_discharge_current=(
            _cumsum_masked("current_average", discharging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_charge_energy_vavg=(
            _cumsum_energy_masked("current_average", "voltage_average", charging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_discharge_energy_vavg=(
            _cumsum_energy_masked("current_average", "voltage_average", discharging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_charge_energy_vcell=(
            _cumsum_energy_masked("current_average", "total_cell_voltage", charging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_discharge_energy_vcell=(
            _cumsum_energy_masked("current_average", "total_cell_voltage", discharging)
            .over("device_serial", order_by="time_stamp")
        ),
        cumul_resting_current=(
            _cumsum_masked("current_average", resting)
            .over("device_serial", order_by="time_stamp")
        ),
    )

    # SOC delta: reference = device with highest mean SOC across the full run
    mean_soc = (
        df.group_by("device_serial")
        .agg(pl.col("soc").mean().alias("mean_soc"))
        .sort("mean_soc", descending=True)
    )
    reference_serial = mean_soc["device_serial"][0]
    logger.info(f"SOC delta reference device (highest mean SOC): {reference_serial}")

    ref_soc = (
        df.filter(pl.col("device_serial") == reference_serial)
        .select("time_stamp", pl.col("soc").alias("ref_soc"))
    )

    df = (
        df.join(ref_soc, on="time_stamp", how="left")
        .with_columns(soc_delta=(pl.col("ref_soc") - pl.col("soc")).abs())
        .drop("ref_soc")
    )

    return df


# ============================================================================
# PLOTTING
# ============================================================================
def _generate_device_colors(num_devices: int) -> list[str]:
    """
    Return *num_devices* distinct hex colour strings.

    Args:
        num_devices: Number of colours required.

    Returns:
        List of hex colour strings of length num_devices.
    """
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if num_devices <= len(palette):
        return palette[:num_devices]

    import colorsys
    logger.warning(f"More devices ({num_devices}) than palette colours — generating extras.")
    colors = palette.copy()
    for i in range(num_devices - len(palette)):
        rgb = colorsys.hsv_to_rgb(i / (num_devices - len(palette)), 0.7, 0.9)
        colors.append("#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in rgb)))
    return colors


def create_figure(
    df: pl.DataFrame,
    chosen_metrics: list[tuple[str, str, str]],
) -> go.Figure:
    """
    Build a Plotly figure containing one subplot per chosen metric.

    Args:
        df:              Full DataFrame with all derived columns.
        chosen_metrics:  List of (column, title, y_label) tuples to plot.

    Returns:
        Plotly Figure ready for display.
    """
    logger.info(f"Building figure for {len(chosen_metrics)} metric(s)")

    device_serials = sorted(df["device_serial"].unique().to_list())
    colors = _generate_device_colors(len(device_serials))
    color_map = dict(zip(device_serials, colors))

    n_rows = len(chosen_metrics)
    titles = [title for _, title, _ in chosen_metrics]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": False}]] * n_rows,
    )

    for row_idx, (col, title, y_label) in enumerate(chosen_metrics, start=1):
        for device_serial in device_serials:
            device_data = (
                df.filter(pl.col("device_serial") == device_serial)
                .sort("time_stamp")
            )
            logger.debug(f"  Row {row_idx} '{col}' / '{device_serial}' — {len(device_data)} pts")

            fig.add_trace(
                go.Scatter(
                    x=device_data["time_stamp"].to_list(),
                    y=device_data[col].to_list(),
                    mode="lines+markers",
                    name=str(device_serial),
                    line=dict(color=color_map[device_serial], width=2),
                    marker=dict(size=4, opacity=0.6),
                    legendgroup=str(device_serial),
                    showlegend=(row_idx == 1),
                    hovertemplate=(
                        f"<b>{device_serial}</b><br>"
                        f"Time: %{{x}}<br>"
                        f"{title}: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

        fig.update_yaxes(title_text=y_label, row=row_idx, col=1)

    fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)
    fig.update_layout(
        title_text="Battery Metrics: Synchronized Device Analysis",
        title_font_size=18,
        height=max(400, n_rows * 280),
        showlegend=True,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        hovermode="x unified",
        template="plotly_white",
    )

    logger.info("Figure built successfully")
    return fig


# ============================================================================
# MAIN
# ============================================================================
def main(data_dir: str = "data") -> None:
    """
    Orchestrate CSV loading, metric derivation, interactive selection, and plotting.

    Args:
        data_dir: Directory containing input CSV files (default: "data").
    """
    logger.info("=" * 80)
    logger.info("Battery Metrics Visualization Script Started")
    logger.info("=" * 80)

    try:
        dataframes = load_all_csvs_from_directory(data_dir)
        if not dataframes:
            logger.error("No dataframes loaded. Exiting.")
            sys.exit(1)

        logger.info(f"Concatenating {len(dataframes)} dataframe(s)")
        df_merged = pl.concat(dataframes)
        logger.info(f"Merged dataframe shape: {df_merged.shape}")

        df_merged = derive_all_metrics(df_merged)
        logger.info(f"Data preview:\n{df_merged.head()}")

        # Interactive metric selection (pre-selects soc, current_average,
        # cumulative_current_average, soc_delta by default)
        chosen_metrics = prompt_metric_selection(preselect=DEFAULT_SELECTION)

        fig = create_figure(df_merged, chosen_metrics)
        logger.info("Rendering figure...")
        fig.show()

        logger.info("=" * 80)
        logger.info("Script completed successfully")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    main(data_dir)
