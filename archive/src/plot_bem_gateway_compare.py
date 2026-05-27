"""
plot_bem_gateway_compare.py
───────────────────────────
Compare multiple BEMs on the same gateway.
Pass one log file per BEM; produces one HTML with three stacked subplots:

  1. Mean cell voltage (mV)   — one trace per BEM
  2. Mean soc_rolling (%)     — one trace per BEM
  3. Reported pack SoC (%)    — one trace per BEM  (BMS value, not estimated)

Shared time x-axis. Each BEM gets a distinct colour.

Usage:
    uv run plot_bem_gateway_compare.py \\
        --input logs/gw-000105-BE20000157-P34-01.log \\
        --input logs/gw-000105-BE20000158-P34-02.log \\
        --input logs/gw-000105-BE20000159-P34-03.log \\
        --output logs/gw-000105_compare.html
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import re
import sys
from pathlib import Path

import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

LOG = logging.getLogger("plot_bem_gw_compare")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "plot_bem_gw_compare.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def load_log(path: Path, num_cells: int) -> pl.DataFrame:
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
    df = df.sort("datestamp")

    # Compute per-row mean cell voltage and mean soc_rolling across all cells
    volt_cols = [f"cellVoltage_{i}" for i in range(1, num_cells + 1)
                 if f"cellVoltage_{i}" in df.columns]
    soc_cols  = [f"soc_rolling_{i}" for i in range(1, num_cells + 1)
                 if f"soc_rolling_{i}" in df.columns]

    df = df.with_columns([
        pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in volt_cols])
          .alias("mean_voltage"),
        pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in soc_cols])
          .alias("mean_soc_rolling"),
    ])
    return df


def label_from_path(path: Path) -> str:
    """Use the filename stem as the legend label; trim common prefix later."""
    return path.stem


def shorten_labels(labels: list[str]) -> list[str]:
    """Strip longest common prefix so long gateway names don't clutter legend."""
    if len(labels) <= 1:
        return labels
    prefix_len = 0
    for chars in zip(*labels):
        if len(set(chars)) == 1:
            prefix_len += 1
        else:
            break
    stripped = [l[prefix_len:] or l for l in labels]
    return stripped


def build_figure(frames: list[tuple[str, pl.DataFrame]]) -> go.Figure:
    palette = pc.qualitative.D3
    colours = [palette[i % len(palette)] for i in range(len(frames))]




    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            "Mean cell voltage (mV)",
            "Mean SoC rolling — estimated (%)",
            "Reported pack SoC (%)",
        ),
    )

    for i, (label, df) in enumerate(frames):
        colour = colours[i]
        x = df["datestamp"].to_list()

        fig.add_trace(go.Scattergl(
            x=x, y=df["mean_voltage"].to_list(),
            name=label, legendgroup=label, mode="lines",
            line=dict(color=colour, width=1.5),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.1f}} mV<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scattergl(
            x=x, y=df["mean_soc_rolling"].to_list(),
            name=label, legendgroup=label, showlegend=False, mode="lines",
            line=dict(color=colour, width=1.5),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.2f}} %<extra></extra>",
        ), row=2, col=1)

        fig.add_trace(go.Scattergl(
            x=x, y=df["soc"].to_list(),
            name=label, legendgroup=label, showlegend=False, mode="lines",
            line=dict(color=colour, width=1.5),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y}} %<extra></extra>",
        ), row=3, col=1)

    fig.update_yaxes(title_text="mV", range=[2700, 3600], row=1, col=1)
    fig.update_yaxes(title_text="%",  range=[0, 100],     row=2, col=1)
    fig.update_yaxes(title_text="%",  range=[0, 100],     row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(
        height=1000,
        hovermode="x unified",
        legend=dict(title="BEM (click to toggle)", tracegroupgap=4),
        margin=dict(l=70, r=160, t=60, b=50),
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", required=True,
                   help="Per-BEM log file (repeat for each BEM)")
    p.add_argument("--output", default=None, help="HTML output path")
    p.add_argument("--num-cells", type=int, default=32)
    p.add_argument("--log-dir", default="./logs")
    args = p.parse_args()

    setup_logging(Path(args.log_dir))

    raw_labels = [Path(f).stem for f in args.input]
    labels = shorten_labels(raw_labels)

    frames: list[tuple[str, pl.DataFrame]] = []
    for label, fpath in zip(labels, args.input):
        path = Path(fpath)
        if not path.exists():
            LOG.error("not found: %s", path); sys.exit(1)
        LOG.info("loading %s → %s", path.name, label)
        frames.append((label, load_log(path, args.num_cells)))

    out_path = (
        Path(args.output) if args.output
        else Path(args.input[0]).parent / "gateway_compare.html"
    )

    fig = build_figure(frames)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
