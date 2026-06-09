
"""
plot_bem_log.py
───────────────
Read one per-BEM CSV log file produced by mqtt_bem_soc_logger.py and
produce ONE HTML report containing 32 per-cell figures.

Each figure shows over time:
    • soc_min, soc_max, soc_rolling   (left y-axis,  0..100 %)
    • cellVoltage                     (right y-axis, 2700..3600 mV)
    • bemCurrentInst                  (far-right y-axis, mA, auto-range)

Optionally animates each figure day-by-day with --animate-by day.

Usage
-----
    uv run plot_bem_log.py \\
        --input logs/gw-000105-BE20000157-P34-02.log
    # writes  logs/gw-000105-BE20000157-P34-02.html
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl


LOG = logging.getLogger("plot_bem_log")

SOC_RANGE = (0, 100)
VOLTAGE_RANGE_MV = (2700, 3600)

# A consistent colour per quantity across all 32 figures
COLOURS = {
    "soc_min":     "#88c0d0",
    "soc_max":     "#a3be8c",
    "soc_rolling": "#bf616a",
    "cellVoltage": "#5e81ac",
    "current":     "#b48ead",
}


# ─── Logging ─────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "plot_bem_log.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


# ─── Load ────────────────────────────────────────────────────────────────────
def load_log(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=2000)

    # Parse the datestamp column to Datetime regardless of input dtype.
    # The logger writes ISO-8601 with a trailing 'Z' (UTC), e.g.
    # "2026-05-22T07:18:58.000Z" — give polars an explicit format.
    if df.schema["datestamp"] != pl.Datetime:
        s = df["datestamp"].cast(pl.Utf8)
        try:
            parsed = s.str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.fZ",
                time_zone="UTC",
                strict=False,
            )
        except Exception:
            parsed = s.str.to_datetime(strict=False)
        df = df.with_columns(datestamp=parsed)

    df = df.sort("datestamp")
    LOG.info("loaded %d rows × %d cols", df.height, df.width)
    return df


# ─── One per-cell figure ─────────────────────────────────────────────────────
def build_cell_figure(
    df: pl.DataFrame,
    cell: int,
    animate_by: str | None,
) -> go.Figure:
    """Static figure (no animation) or animated by day-bucket."""
    full_x = df["sequence_number"].to_list()

    cols = {
        "soc_min":     f"soc_min_{cell}",
        "soc_max":     f"soc_max_{cell}",
        "soc_rolling": f"soc_rolling_{cell}",
        "cellVoltage": f"cellVoltage_{cell}",
        "current":     "bemCurrentInst",
    }

    def make_traces(slice_df: pl.DataFrame) -> list[go.Scattergl]:
        x = slice_df["sequence_number"].to_list()
        return [
            go.Scattergl(x=x, y=slice_df[cols["soc_min"]].to_list(),
                         name="soc_min", yaxis="y", mode="lines",
                         line=dict(color=COLOURS["soc_min"], width=1, dash="dot")),
            go.Scattergl(x=x, y=slice_df[cols["soc_max"]].to_list(),
                         name="soc_max", yaxis="y", mode="lines",
                         line=dict(color=COLOURS["soc_max"], width=1, dash="dot")),
            go.Scattergl(x=x, y=slice_df[cols["soc_rolling"]].to_list(),
                         name="soc_rolling", yaxis="y", mode="lines",
                         line=dict(color=COLOURS["soc_rolling"], width=2.2)),
            go.Scattergl(x=x, y=slice_df[cols["cellVoltage"]].to_list(),
                         name="cellVoltage (mV)", yaxis="y2", mode="lines",
                         line=dict(color=COLOURS["cellVoltage"], width=1.2)),
            go.Scattergl(x=x, y=slice_df[cols["current"]].to_list(),
                         name="bemCurrentInst (mA)", yaxis="y3", mode="lines",
                         line=dict(color=COLOURS["current"], width=1, dash="dash"),
                         opacity=0.7),
        ]

    # Build the figure (initial traces use either full data or first frame)
    if not animate_by:
        fig = go.Figure(data=make_traces(df))
        frames = None
    else:
        if animate_by == "day":
            bucket_expr = pl.col("datestamp").dt.truncate("1d").alias("_bucket")
        elif animate_by == "hour":
            bucket_expr = pl.col("datestamp").dt.truncate("1h").alias("_bucket")
        else:
            raise ValueError(f"unknown --animate-by: {animate_by!r}")

        df_b = df.with_columns(bucket_expr)
        buckets = sorted(df_b["_bucket"].unique().to_list())
        first = df_b.filter(pl.col("_bucket") == buckets[0])

        fig = go.Figure(data=make_traces(first))
        frames = [
            go.Frame(
                name=str(b),
                data=make_traces(df_b.filter(pl.col("_bucket") == b)),
            )
            for b in buckets
        ]
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=500, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                y=-0.25, x=0.05, len=0.9,
                currentvalue=dict(prefix=f"{animate_by}: "),
                steps=[
                    dict(method="animate", label=str(b),
                         args=[[str(b)],
                               dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True))])
                    for b in buckets
                ],
            )],
        )

    # Fixed axes per user spec; current axis auto-ranges from the data.
    fig.update_layout(
        title=f"Cell {cell}",
        height=440,
        margin=dict(l=60, r=140, t=60, b=90 if animate_by else 60),
        xaxis=dict(domain=[0.06, 0.86], title="Time"),
        yaxis=dict(title="SoC (%)", range=list(SOC_RANGE), side="left",
                   showgrid=True, zeroline=False),
        yaxis2=dict(title="Voltage (mV)", range=list(VOLTAGE_RANGE_MV),
                    overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Current (mA)", overlaying="y", side="right",
                    anchor="free", position=0.96, showgrid=False),
        legend=dict(orientation="h", y=-0.18 if not animate_by else -0.38),
        hovermode="x unified",
    )
    return fig


# ─── Write one HTML containing all 32 figures ────────────────────────────────
def write_html(figs: list[go.Figure], title: str, out_path: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "  body { font-family: -apple-system, system-ui, sans-serif; ",
        "         margin: 24px; background: #fafafa; }",
        "  h1 { font-size: 20px; }",
        "  .cell-block { background: white; padding: 8px 12px; margin: 16px 0;",
        "                border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for cell, fig in enumerate(figs, start=1):
        parts.append("<div class='cell-block'>")
        parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        parts.append("</div>")
    parts.append("</body></html>")

    out_path.write_text("".join(parts), encoding="utf-8")


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Per-BEM log file (CSV)")
    p.add_argument("--output", default=None,
                   help="HTML output path (default: input with .html suffix)")
    p.add_argument("--num-cells", type=int, default=32)
    p.add_argument("--animate-by", choices=["day", "hour"], default=None,
                   help="If set, animate each per-cell figure by that bucket.")
    p.add_argument("--log-dir", default="./logs",
                   help="Where to write the script log file.")
    args = p.parse_args()

    setup_logging(Path(args.log_dir))

    in_path = Path(args.input)
    if not in_path.exists():
        LOG.error("input not found: %s", in_path)
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".html")

    LOG.info("reading %s", in_path)
    df = load_log(in_path)

    LOG.info("building %d per-cell figures (animate_by=%s) …",
             args.num_cells, args.animate_by)
    figs = [
        build_cell_figure(df, c, args.animate_by)
        for c in range(1, args.num_cells + 1)
    ]

    LOG.info("writing %s", out_path)
    write_html(figs, title=in_path.name, out_path=out_path)
    LOG.info("done: %s", out_path)


if __name__ == "__main__":
    main()
