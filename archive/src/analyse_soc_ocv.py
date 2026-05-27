# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "plotly"]
# ///
"""
analyse_soc_ocv.py
──────────────────
Reads the output CSV produced by soc_sequential_ocv.py and evaluates how
well the OCV LUT resets track the reported SoC (soc_nominal).

Only rows where mode == "LUT" are scored — CC, UNINIT, and SKIP rows are
excluded from all error metrics.

Metrics computed
────────────────
  MAE   — Mean Absolute Error  (%)
  RMSE  — Root Mean Squared Error  (%)
  MAX   — Maximum absolute error  (%)
  BIAS  — Mean signed error  (soc_mid - soc_nominal)  (%)
  N     — Number of LUT reset events scored

Reported at two levels:
  • Per-cell
  • Per-mode (aggregate across all cells — only LUT here)

Plots (Plotly, opens in browser)
────────────────────────────────
  1. SoC over time  — soc_mid (LUT resets only, markers) overlaid on
                      soc_nominal (line) for each selected cell in a
                      subplot grid.

  2. Absolute error over time  — |soc_mid - soc_nominal| for each
                                  selected cell, subplot grid.

  3. Error distribution  — histogram of (soc_mid - soc_nominal) across
                           all selected cells combined.

  4. Per-cell summary bar chart  — MAE per cell.

Usage
─────
uv run analyse_soc_ocv.py \\
    --csv          soc_ocv_output.csv \\
    --cells        1 2 3 4          # space-separated cell indices to plot
    --log_file     analyse_soc_ocv.log

Optional:
    --save_html    results.html     # if set, also saves plots to this file
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("analyse_soc")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(errors: list[float]) -> dict:
    """errors = list of (soc_mid - soc_nominal) signed values."""
    if not errors:
        return {"n": 0, "mae": None, "rmse": None, "max_abs": None, "bias": None}
    n    = len(errors)
    mae  = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    bias = sum(errors) / n
    max_abs = max(abs(e) for e in errors)
    return {"n": n, "mae": mae, "rmse": rmse, "max_abs": max_abs, "bias": bias}


def fmt(val: float | None, dp: int = 3) -> str:
    return f"{val:.{dp}f}" if val is not None else "—"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",      required=True, help="Output CSV from soc_sequential_ocv.py")
    p.add_argument(
        "--cells", nargs="+", type=int, default=None,
        help="Cell indices to plot (e.g. --cells 1 2 3). Default: all cells.",
    )
    p.add_argument("--save_html", default=None, help="Optional path to save combined HTML")
    p.add_argument("--log_file",  default="analyse_soc_ocv.log")
    args = p.parse_args()

    log = setup_logging(args.log_file)
    log.info("──── analyse_soc_ocv start ────")
    log.info("Args: %s", vars(args))

    if not Path(args.csv).exists():
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading %s", args.csv)
    df = pl.read_csv(args.csv, try_parse_dates=False)
    log.info("Rows: %d  Columns: %s", df.height, df.columns)

    required = {"sequence_number", "cell_idx", "mode", "soc_mid", "soc_nominal", "time_stamp"}
    missing  = required - set(df.columns)
    if missing:
        log.error("CSV missing columns: %s", sorted(missing))
        sys.exit(1)

    all_cells = sorted(df["cell_idx"].unique().to_list())
    log.info("Cells found in CSV: %s", all_cells)

    plot_cells = args.cells if args.cells else all_cells
    invalid = [c for c in plot_cells if c not in all_cells]
    if invalid:
        log.warning("Requested cells not in CSV (ignored): %s", invalid)
        plot_cells = [c for c in plot_cells if c in all_cells]
    if not plot_cells:
        log.error("No valid cells to plot.")
        sys.exit(1)
    log.info("Plotting cells: %s", plot_cells)

    # ── Cast numeric columns (CSV read may produce Utf8 for nullable floats) ──
    for col in ("soc_mid", "soc_max", "soc_min", "soc_hybrid", "soc_nominal"):
        if col in df.columns and df[col].dtype != pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    # ── Filter to LUT rows only for scoring ───────────────────────────────────
    lut_df = df.filter(pl.col("mode") == "LUT")
    log.info("LUT reset rows: %d / %d total", lut_df.height, df.height)

    if lut_df.is_empty():
        log.error("No LUT reset rows found in CSV — nothing to score.")
        sys.exit(1)

    lut_df = lut_df.with_columns(
        error=(pl.col("soc_mid").cast(pl.Float64) - pl.col("soc_nominal").cast(pl.Float64))
    )

    # ── Per-cell metrics ──────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 65)
    log.info("  PER-CELL METRICS  (mode=LUT only,  soc_mid vs soc_nominal)")
    log.info("═" * 65)
    log.info("  %-6s  %6s  %7s  %7s  %8s  %8s", "Cell", "N", "MAE %", "RMSE %", "MaxAbs %", "Bias %")
    log.info("  " + "─" * 60)

    cell_metrics: dict[int, dict] = {}
    for cell in all_cells:
        cell_lut = lut_df.filter(pl.col("cell_idx") == cell)
        errs = cell_lut["error"].drop_nulls().to_list()
        m = compute_metrics(errs)
        cell_metrics[cell] = m
        log.info(
            "  cell %-3d  %6d  %7s  %7s  %8s  %8s",
            cell, m["n"],
            fmt(m["mae"]), fmt(m["rmse"]), fmt(m["max_abs"]), fmt(m["bias"]),
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_errors = lut_df["error"].drop_nulls().to_list()
    agg = compute_metrics(all_errors)
    log.info("  " + "─" * 60)
    log.info(
        "  %-6s  %6d  %7s  %7s  %8s  %8s",
        "ALL", agg["n"],
        fmt(agg["mae"]), fmt(agg["rmse"]), fmt(agg["max_abs"]), fmt(agg["bias"]),
    )
    log.info("═" * 65)

    # Print to stdout too so it's visible without tailing the log
    print()
    print("═" * 65)
    print("  PER-CELL METRICS  (mode=LUT only,  soc_mid vs soc_nominal)")
    print("═" * 65)
    print(f"  {'Cell':<6}  {'N':>6}  {'MAE %':>7}  {'RMSE %':>7}  {'MaxAbs %':>8}  {'Bias %':>8}")
    print("  " + "─" * 60)
    for cell in all_cells:
        m = cell_metrics[cell]
        print(
            f"  cell {cell:<3}  {m['n']:>6}  {fmt(m['mae']):>7}  {fmt(m['rmse']):>7}"
            f"  {fmt(m['max_abs']):>8}  {fmt(m['bias']):>8}"
        )
    print("  " + "─" * 60)
    print(
        f"  {'ALL':<6}  {agg['n']:>6}  {fmt(agg['mae']):>7}  {fmt(agg['rmse']):>7}"
        f"  {fmt(agg['max_abs']):>8}  {fmt(agg['bias']):>8}"
    )
    print("═" * 65)
    print()

    # ── Plotting helpers ──────────────────────────────────────────────────────
    COLS   = min(3, len(plot_cells))
    ROWS   = math.ceil(len(plot_cells) / COLS)
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    def cell_color(i: int) -> str:
        return COLORS[i % len(COLORS)]

    def subplot_titles(label: str) -> list[str]:
        return [f"Cell {c} — {label}" for c in plot_cells]

    # Pre-split data per cell for speed
    cell_data: dict[int, pl.DataFrame] = {}
    cell_lut_data: dict[int, pl.DataFrame] = {}
    for cell in plot_cells:
        cell_data[cell]     = df.filter(pl.col("cell_idx") == cell).sort("sequence_number")
        cell_lut_data[cell] = lut_df.filter(pl.col("cell_idx") == cell).sort("sequence_number")

    # ── Plot 1: SoC over time ─────────────────────────────────────────────────
    log.info("Building Plot 1: SoC over time")
    fig1 = make_subplots(
        rows=ROWS, cols=COLS,
        subplot_titles=subplot_titles("SoC over time"),
        shared_xaxes=False,
        vertical_spacing=0.08,
    )
    for i, cell in enumerate(plot_cells):
        row, col = divmod(i, COLS)
        row += 1; col += 1
        color = cell_color(i)
        cd  = cell_data[cell]
        cld = cell_lut_data[cell]

        # nominal line (all rows — gives context even between resets)
        fig1.add_trace(go.Scatter(
            x=cd["sequence_number"].to_list(),
            y=cd["soc_nominal"].to_list(),
            mode="lines",
            name=f"Cell {cell} nominal",
            line=dict(color=color, width=1.5, dash="dot"),
            legendgroup=f"cell{cell}",
            showlegend=(i == 0),
        ), row=row, col=col)

        # LUT resets — markers only
        fig1.add_trace(go.Scatter(
            x=cld["sequence_number"].to_list(),
            y=cld["soc_mid"].to_list(),
            mode="markers",
            name=f"Cell {cell} OCV LUT",
            marker=dict(color=color, size=6, symbol="circle-open", line=dict(width=1.5)),
            legendgroup=f"cell{cell}_lut",
            showlegend=(i == 0),
        ), row=row, col=col)

        fig1.update_xaxes(title_text="Sequence", row=row, col=col)
        fig1.update_yaxes(title_text="SoC (%)", row=row, col=col)

    fig1.update_layout(
        title="OCV LUT SoC vs Nominal SoC",
        height=350 * ROWS + 100,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05),
    )

    # ── Plot 2: Absolute error over time ──────────────────────────────────────
    log.info("Building Plot 2: Absolute error over time")
    fig2 = make_subplots(
        rows=ROWS, cols=COLS,
        subplot_titles=subplot_titles("|Error| over time"),
        vertical_spacing=0.08,
    )
    for i, cell in enumerate(plot_cells):
        row, col = divmod(i, COLS)
        row += 1; col += 1
        color = cell_color(i)
        cld = cell_lut_data[cell]
        abs_err = [abs(e) for e in cld["error"].to_list()]

        fig2.add_trace(go.Scatter(
            x=cld["sequence_number"].to_list(),
            y=abs_err,
            mode="markers+lines",
            name=f"Cell {cell}",
            marker=dict(color=color, size=5),
            line=dict(color=color, width=1),
            legendgroup=f"cell{cell}",
            showlegend=True,
        ), row=row, col=col)

        # MAE reference line
        mae = cell_metrics[cell]["mae"]
        if mae is not None and len(cld) > 0:
            fig2.add_hline(
                y=mae, line_dash="dash", line_color=color, opacity=0.5,
                annotation_text=f"MAE={mae:.2f}%",
                annotation_position="top right",
                row=row, col=col,
            )

        fig2.update_xaxes(title_text="Sequence", row=row, col=col)
        fig2.update_yaxes(title_text="|Error| (%)", row=row, col=col)

    fig2.update_layout(
        title="OCV LUT Absolute Error over Time",
        height=350 * ROWS + 100,
        template="plotly_white",
    )

    # ── Plot 3: Error distribution (histogram) ────────────────────────────────
    log.info("Building Plot 3: Error distribution")
    fig3 = go.Figure()
    combined_errors: list[float] = []
    for i, cell in enumerate(plot_cells):
        errs = cell_lut_data[cell]["error"].drop_nulls().to_list()
        combined_errors.extend(errs)
        fig3.add_trace(go.Histogram(
            x=errs,
            name=f"Cell {cell}",
            opacity=0.6,
            nbinsx=40,
            marker_color=cell_color(i),
        ))
    fig3.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    fig3.update_layout(
        title="Error Distribution  (soc_mid − soc_nominal, LUT resets only)",
        xaxis_title="Error (%)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=500,
    )

    # ── Plot 4: Per-cell MAE bar chart ────────────────────────────────────────
    log.info("Building Plot 4: Per-cell MAE bar chart")
    fig4 = go.Figure()
    bar_cells = plot_cells
    mae_vals  = [cell_metrics[c]["mae"]  for c in bar_cells]
    rmse_vals = [cell_metrics[c]["rmse"] for c in bar_cells]
    bias_vals = [cell_metrics[c]["bias"] for c in bar_cells]
    n_vals    = [cell_metrics[c]["n"]    for c in bar_cells]

    fig4.add_trace(go.Bar(
        x=[f"Cell {c}" for c in bar_cells],
        y=mae_vals,
        name="MAE",
        marker_color="#1f77b4",
        text=[f"{v:.3f}%" if v is not None else "—" for v in mae_vals],
        textposition="outside",
    ))
    fig4.add_trace(go.Bar(
        x=[f"Cell {c}" for c in bar_cells],
        y=rmse_vals,
        name="RMSE",
        marker_color="#ff7f0e",
        text=[f"{v:.3f}%" if v is not None else "—" for v in rmse_vals],
        textposition="outside",
    ))
    fig4.add_trace(go.Bar(
        x=[f"Cell {c}" for c in bar_cells],
        y=[abs(b) if b is not None else None for b in bias_vals],
        name="|Bias|",
        marker_color="#2ca02c",
        text=[f"{v:.3f}%" if v is not None else "—" for v in bias_vals],
        textposition="outside",
    ))
    fig4.update_layout(
        title="Per-Cell Error Metrics  (LUT resets only)",
        xaxis_title="Cell",
        yaxis_title="Error (%)",
        barmode="group",
        template="plotly_white",
        height=500,
        xaxis=dict(tickangle=-45),
    )

    # ── Show in browser ───────────────────────────────────────────────────────
    log.info("Opening plots in browser")
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()

    # ── Optional save ─────────────────────────────────────────────────────────
    if args.save_html:
        import plotly.io as pio
        log.info("Saving combined HTML to %s", args.save_html)
        combined_html = (
            fig1.to_html(full_html=True, include_plotlyjs="cdn")
            + "\n"
            + fig2.to_html(full_html=False, include_plotlyjs=False)
            + "\n"
            + fig3.to_html(full_html=False, include_plotlyjs=False)
            + "\n"
            + fig4.to_html(full_html=False, include_plotlyjs=False)
        )
        Path(args.save_html).write_text(combined_html, encoding="utf-8")
        log.info("Saved.")

    log.info("──── analyse_soc_ocv done ────")


if __name__ == "__main__":
    main()
