
"""
plot_bem_compare_cells.py
─────────────────────────
One HTML, two stacked subplots sharing the time x-axis:

  • Top   — cellVoltage_1 … cellVoltage_32   (mV, range 2700..3600)
  • Bottom — soc_rolling_1 … soc_rolling_32  (%, range 0..100)
             plus reported pack `soc` as a thick black line

Cells are coloured along a viridis ramp (cell 1 → dark, cell 32 → light)
so drift between adjacent cells is easy to spot. Click any legend entry to
hide/show that cell; double-click to isolate.

Usage:
    uv run plot_bem_compare_cells.py --input logs/gw-000105-BE20000157-P34-02.log
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

LOG = logging.getLogger("plot_bem_compare")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "plot_bem_compare.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def load_log(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=2000)
    if df.schema["datestamp"] != pl.Datetime:
        s = df["datestamp"].cast(pl.Utf8)
        try:
            parsed = s.str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC", strict=False,
            )
        except Exception:
            parsed = s.str.to_datetime(strict=False)
        df = df.with_columns(datestamp=parsed)
    return df.sort("datestamp")


def cell_colour(cell: int, num_cells: int) -> str:
    """Pick a colour along the viridis ramp for cell index 1..num_cells."""
    t = (cell - 1) / max(num_cells - 1, 1)
    return pc.sample_colorscale("Viridis", [t])[0]


def build_figure(df: pl.DataFrame, num_cells: int) -> go.Figure:
    x = df["sequence_number"].to_list()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Cell voltage (mV)", "SoC rolling (%) — black = reported pack SoC"),
    )

    for cell in range(1, num_cells + 1):
        colour = cell_colour(cell, num_cells)
        # Voltage (top)
        fig.add_trace(
            go.Scattergl(
                x=x, y=df[f"cellVoltage_{cell}"].to_list(),
                name=f"cell {cell:02d}", legendgroup=f"cell{cell}",
                mode="lines", line=dict(color=colour, width=1),
                hovertemplate=f"cell {cell:02d}<br>%{{x}}<br>%{{y}} mV<extra></extra>",
            ),
            row=1, col=1,
        )
        # SoC rolling (bottom)
        fig.add_trace(
            go.Scattergl(
                x=x, y=df[f"soc_rolling_{cell}"].to_list(),
                name=f"cell {cell:02d}", legendgroup=f"cell{cell}",
                mode="lines", line=dict(color=colour, width=1),
                showlegend=False,  # legend already shown for voltage trace
                hovertemplate=f"cell {cell:02d}<br>%{{x}}<br>%{{y:.2f}} %<extra></extra>",
            ),
            row=2, col=1,
        )

    # Reported pack SoC overlaid on the bottom subplot
    fig.add_trace(
        go.Scattergl(
            x=x, y=df["soc"].to_list(),
            name="reported soc",
            mode="lines", line=dict(color="black", width=2.5),
            hovertemplate="reported<br>%{x}<br>%{y} %<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="mV", range=[2700, 3600], row=1, col=1)
    fig.update_yaxes(title_text="%",  range=[0, 100],    row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_layout(
        height=900,
        hovermode="x unified",
        legend=dict(
            title="Cell (click to toggle)",
            tracegroupgap=2,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=160, t=60, b=50),
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Per-BEM log file (CSV)")
    p.add_argument("--output", default=None, help="HTML output path")
    p.add_argument("--num-cells", type=int, default=32)
    p.add_argument("--log-dir", default="./logs")
    args = p.parse_args()

    setup_logging(Path(args.log_dir))
    in_path = Path(args.input)
    if not in_path.exists():
        LOG.error("input not found: %s", in_path); sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_compare.html"
    )

    LOG.info("reading %s", in_path)
    df = load_log(in_path)
    LOG.info("loaded %d rows × %d cols", df.height, df.width)

    fig = build_figure(df, args.num_cells)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
