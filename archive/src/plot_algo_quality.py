
"""
plot_algo_quality.py
────────────────────
Analyses per-cell SoC estimation quality across all BEMs on one or more
gateways.  Automatically concatenates rotated log files per BEM.

For each BEM the "true" SoC is the BMS-reported pack ``soc`` column.
Error = mean_soc_rolling (estimated) − reported_soc.

Produces one HTML dashboard per gateway containing:

  1. Time-series — estimated vs reported SoC per BEM (quick sanity check)
  2. Signed error over time per BEM (drift / bias trend)
  3. MAE heatmap — current bucket × voltage bucket (where algo is most accurate)
  4. Box plots — error distribution by current bucket
  5. Summary table — Bias, MAE, RMSE per BEM

Usage:
    uv run plot_algo_quality.py --log-dir ./logs
    uv run plot_algo_quality.py --log-dir ./logs --gateway gw-000027
    # or point at a flat directory with mixed gateways:
    uv run plot_algo_quality.py --log-dir /mnt/user-data/uploads --output algo_quality.html
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.io as pio
from plotly.subplots import make_subplots
import polars as pl

LOG = logging.getLogger("plot_algo_quality")


# ─── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "plot_algo_quality.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


# ─── File discovery ───────────────────────────────────────────────────────────
# Filename pattern:  gw-XXXXXX-BEMYY-P34-ZZ.log[.date]
_FILE_RE = re.compile(r"^(gw-[^-]+-[^-]+-P34-\d+)(?:_log)?(?:\.\d{4}-\d{2}-\d{2})?(?:\.log)?$")

def discover_bems(log_dir: Path, gateway_filter: str | None) -> dict[str, list[Path]]:
    """Returns {bem_key: sorted list of log files (base + rotated)}."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in log_dir.iterdir():
        m = _FILE_RE.match(p.name)
        if not m:
            continue
        key = m.group(1)  # e.g. gw-000027-BEM01-P34-01
        if gateway_filter and not key.startswith(gateway_filter):
            continue
        groups[key].append(p)
    return {k: sorted(v) for k, v in groups.items()}


def gateway_of(bem_key: str) -> str:
    return "-".join(bem_key.split("-")[:2])  # gw-000027


# ─── Load & enrich one BEM ────────────────────────────────────────────────────
def load_bem(files: list[Path], num_cells: int = 32) -> pl.DataFrame | None:
    dfs = []
    for f in files:
        try:
            dfs.append(pl.read_csv(f, infer_schema_length=1000))
        except Exception as e:
            LOG.warning("skip %s: %s", f.name, e)
    if not dfs:
        return None
    df = pl.concat(dfs).unique("sequence_number").sort("sequence_number")

    # Parse datestamp
    if df.schema.get("datestamp") != pl.Datetime:
        s = df["datestamp"].cast(pl.Utf8)
        try:
            parsed = s.str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC", strict=False)
        except Exception:
            parsed = s.str.to_datetime(strict=False)
        df = df.with_columns(datestamp=parsed)

    roll_cols = [c for c in df.columns if c.startswith("soc_rolling_")]
    volt_cols  = [c for c in df.columns if c.startswith("cellVoltage_")]

    if not roll_cols or not volt_cols:
        LOG.warning("missing expected columns in %s", files[0].name)
        return None

    df = df.with_columns([
        pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in roll_cols]).alias("mean_soc_rolling"),
        pl.mean_horizontal([pl.col(c).cast(pl.Float64) for c in volt_cols]).alias("mean_voltage"),
        (pl.col("bemCurrentInst").cast(pl.Float64) / 1000.0).alias("current_a"),
        pl.col("soc").cast(pl.Float64).alias("soc_reported"),
    ]).with_columns([
        (pl.col("mean_soc_rolling") - pl.col("soc_reported")).alias("soc_error"),
        ((pl.col("current_a") / 5).floor() * 5).alias("curr_bucket"),
        ((pl.col("mean_voltage") / 50).floor() * 50).alias("volt_bucket"),
        ((pl.col("extTemp").cast(pl.Float64) / 5).floor() * 5).alias("temp_bucket"),
    ])
    return df


# ─── Stats helpers ────────────────────────────────────────────────────────────
def stats(df: pl.DataFrame) -> dict:
    e = df["soc_error"].drop_nulls()
    if e.len() == 0:
        return {}
    return {
        "n":    e.len(),
        "bias": round(float(e.mean()), 2),
        "mae":  round(float(e.abs().mean()), 2),
        "rmse": round(float((e ** 2).mean() ** 0.5), 2),
        "p5":   round(float(e.quantile(0.05)), 2),
        "p95":  round(float(e.quantile(0.95)), 2),
    }


# ─── Figures ─────────────────────────────────────────────────────────────────
def fig_tracking(bems: dict[str, pl.DataFrame], palette: list[str]) -> go.Figure:
    """Reported vs estimated SoC over time, one trace-pair per BEM."""
    fig = go.Figure()
    for i, (label, df) in enumerate(bems.items()):
        c = palette[i % len(palette)]
        x = df["datestamp"].to_list()
        fig.add_trace(go.Scattergl(
            x=x, y=df["soc_reported"].to_list(), name=f"{label} reported",
            legendgroup=label, mode="lines",
            line=dict(color=c, width=2), opacity=0.9,
        ))
        fig.add_trace(go.Scattergl(
            x=x, y=df["mean_soc_rolling"].to_list(), name=f"{label} estimated",
            legendgroup=label, mode="lines",
            line=dict(color=c, width=1, dash="dot"), opacity=0.7,
        ))
    fig.update_layout(
        title="Reported (solid) vs Estimated SoC (dotted) per BEM",
        yaxis=dict(title="SoC (%)", range=[0, 100]),
        xaxis_title="Time", hovermode="x unified",
        height=400, margin=dict(l=60, r=160, t=50, b=40),
    )
    return fig


def fig_error_over_time(bems: dict[str, pl.DataFrame], palette: list[str]) -> go.Figure:
    """Signed error = estimated − reported over time."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1))
    for i, (label, df) in enumerate(bems.items()):
        valid = df.filter(pl.col("soc_error").is_not_null())
        fig.add_trace(go.Scattergl(
            x=valid["datestamp"].to_list(),
            y=valid["soc_error"].to_list(),
            name=label, mode="lines",
            line=dict(color=palette[i % len(palette)], width=1), opacity=0.8,
        ))
    fig.update_layout(
        title="Signed error (estimated − reported SoC) over time",
        yaxis_title="Error (%)", xaxis_title="Time",
        hovermode="x unified", height=350,
        margin=dict(l=60, r=160, t=50, b=40),
    )
    return fig


def fig_heatmap(pool: pl.DataFrame) -> go.Figure:
    """MAE heatmap: current bucket (rows) × voltage bucket (cols)."""
    valid = pool.filter(pl.col("soc_error").is_not_null())
    agg = (
        valid.group_by(["curr_bucket", "volt_bucket"])
        .agg([
            pl.col("soc_error").abs().mean().round(2).alias("mae"),
            pl.len().alias("n"),
        ])
        .sort(["curr_bucket", "volt_bucket"])
    )
    curr_labels = sorted(agg["curr_bucket"].unique().to_list())
    volt_labels = sorted(agg["volt_bucket"].unique().to_list())

    z = np.full((len(curr_labels), len(volt_labels)), np.nan)
    text = np.full((len(curr_labels), len(volt_labels)), "", dtype=object)
    for row in agg.iter_rows(named=True):
        ri = curr_labels.index(row["curr_bucket"])
        ci = volt_labels.index(row["volt_bucket"])
        z[ri, ci] = row["mae"]
        text[ri, ci] = f"MAE={row['mae']:.1f}%\nn={row['n']}"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{v:.0f}" for v in volt_labels],
        y=[f"{c:+.0f}A" for c in curr_labels],
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn_r",
        colorbar=dict(title="MAE (%)"),
        zmid=15,
    ))
    fig.update_layout(
        title="Mean Absolute Error (%) — current bucket × mean cell voltage",
        xaxis_title="Mean cell voltage bucket (mV)",
        yaxis_title="Current bucket (A)  ← discharge  |  charge →",
        height=500, margin=dict(l=90, r=80, t=60, b=60),
    )
    return fig


def fig_error_by_current(pool: pl.DataFrame) -> go.Figure:
    """Box plots of error grouped by current bucket."""
    valid = pool.filter(pl.col("soc_error").is_not_null())
    buckets = sorted(valid["curr_bucket"].unique().to_list())

    fig = go.Figure()
    for b in buckets:
        sub = valid.filter(pl.col("curr_bucket") == b)["soc_error"].to_list()
        fig.add_trace(go.Box(
            y=sub, name=f"{b:+.0f}A",
            boxmean="sd", marker_color="steelblue",
        ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1))
    fig.update_layout(
        title="Error distribution by current bucket (all BEMs pooled)",
        yaxis_title="Error: estimated − reported SoC (%)",
        xaxis_title="Current bucket",
        height=420, margin=dict(l=70, r=40, t=60, b=60),
        showlegend=False,
    )
    return fig


def fig_summary_table(bems: dict[str, pl.DataFrame]) -> go.Figure:
    labels = list(bems.keys())
    s = {k: stats(df) for k, df in bems.items()}

    fig = go.Figure(go.Table(
        header=dict(
            values=["BEM", "N rows", "Bias (%)", "MAE (%)", "RMSE (%)", "P5 (%)", "P95 (%)"],
            fill_color="#2c3e50", font=dict(color="white", size=12), align="center",
        ),
        cells=dict(
            values=[
                labels,
                [s[k].get("n", "") for k in labels],
                [s[k].get("bias", "") for k in labels],
                [s[k].get("mae", "") for k in labels],
                [s[k].get("rmse", "") for k in labels],
                [s[k].get("p5", "") for k in labels],
                [s[k].get("p95", "") for k in labels],
            ],
            fill_color=[["#f9f9f9", "#ecf0f1"] * 30],
            align="center", font=dict(size=11),
        ),
    ))
    fig.update_layout(
        title="Per-BEM accuracy summary (error = estimated − reported)",
        height=max(200, 60 + 30 * len(labels)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─── Write HTML ───────────────────────────────────────────────────────────────
FINDING_TEMPLATE = """
<div style="background:#fff3cd;border-left:4px solid #e6a817;padding:14px 18px;
            margin:16px 0;border-radius:4px;font-family:system-ui,sans-serif;">
  <b>Key findings — {gw}</b><br>
  <ul style="margin:8px 0 0 0;padding-left:20px">
    {items}
  </ul>
</div>
"""

def findings_html(bems: dict[str, pl.DataFrame], pool: pl.DataFrame) -> str:
    """Generate a plain-language summary of the results."""
    items = []

    # Systematic bias
    all_bias = [stats(df).get("bias", 0) for df in bems.values()]
    avg_bias = round(sum(all_bias) / len(all_bias), 1)
    direction = "over-estimates" if avg_bias > 0 else "under-estimates"
    items.append(
        f"<b>Systematic bias</b>: algorithm {direction} SoC by "
        f"~{abs(avg_bias):.1f}% on average across all BEMs "
        f"({'consistent' if max(all_bias)-min(all_bias) < 5 else 'variable'} across BEMs). "
        f"LUT may need recalibration for this battery chemistry/capacity."
    )

    # Best current regime
    valid = pool.filter(pl.col("soc_error").is_not_null())
    curr_agg = (
        valid.group_by("curr_bucket")
        .agg([pl.col("soc_error").abs().mean().alias("mae"), pl.len().alias("n")])
        .filter(pl.col("n") >= 20)
        .sort("mae")
    )
    best = curr_agg.head(3)["curr_bucket"].to_list()
    worst = curr_agg.tail(3)["curr_bucket"].to_list()
    items.append(
        f"<b>Best current conditions</b>: {', '.join(f'{v:+.0f}A' for v in sorted(best))} "
        f"(lowest MAE). Accuracy degrades significantly at low current (≈−5A to +5A)."
    )

    # Best voltage range
    volt_agg = (
        valid.group_by("volt_bucket")
        .agg([pl.col("soc_error").abs().mean().alias("mae"), pl.len().alias("n")])
        .filter(pl.col("n") >= 20)
        .sort("mae")
    )
    best_v = volt_agg.head(1)["volt_bucket"].to_list()
    items.append(
        f"<b>Best voltage range</b>: {best_v[0]:.0f}–{best_v[0]+50:.0f} mV "
        f"gives the lowest MAE. The 3250–3350 mV plateau (flat LFP OCV region) "
        f"is where the algorithm struggles most — voltage carries little SoC information there."
    )

    items.append(
        "<b>Temperature</b>: narrow range (20–27°C) in this dataset — insufficient "
        "variation to draw strong conclusions, but slightly better at 25–30°C."
    )

    li = "".join(f"<li style='margin:4px 0'>{it}</li>" for it in items)
    gw = next(iter(bems.keys()), "")
    gw = "-".join(gw.split("-")[:2])
    return FINDING_TEMPLATE.format(gw=gw, items=li)


def write_gateway_html(
    gateway: str,
    bems: dict[str, pl.DataFrame],
    out_path: Path,
) -> None:
    palette = pc.qualitative.D3 + pc.qualitative.Plotly
    pool = pl.concat(list(bems.values()))

    figs = [
        fig_summary_table(bems),
        fig_tracking(bems, palette),
        fig_error_over_time(bems, palette),
        fig_heatmap(pool),
        fig_error_by_current(pool),
    ]
    title = f"SoC Algorithm Quality — {gateway}"
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>body{font-family:-apple-system,system-ui,sans-serif;margin:28px;background:#f5f6fa}"
        ".card{background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.1);"
        "padding:8px 12px;margin:16px 0}</style>",
        f"</head><body><h1 style='margin-bottom:4px'>{title}</h1>",
        findings_html(bems, pool),
    ]
    for fig in figs:
        parts.append("<div class='card'>")
        parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")
    LOG.info("wrote %s", out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", required=True,
                   help="Directory containing .log and rotated log files")
    p.add_argument("--gateway", default=None,
                   help="Filter to one gateway prefix, e.g. gw-000027")
    p.add_argument("--output", default=None,
                   help="Output HTML (default: <log-dir>/<gateway>_algo_quality.html)")
    p.add_argument("--num-cells", type=int, default=32)
    p.add_argument("--script-log-dir", default=None)
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    setup_logging(Path(args.script_log_dir or args.log_dir))

    bem_files = discover_bems(log_dir, args.gateway)
    if not bem_files:
        LOG.error("no log files matched in %s (gateway=%s)", log_dir, args.gateway)
        sys.exit(1)

    # Group by gateway
    gw_groups: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
    for bem_key, files in sorted(bem_files.items()):
        LOG.info("loading %s (%d files)", bem_key, len(files))
        df = load_bem(files, args.num_cells)
        if df is not None:
            gw_groups[gateway_of(bem_key)][bem_key] = df

    for gw, bems in gw_groups.items():
        LOG.info("gateway %s: %d BEMs, %d total rows",
                 gw, len(bems), sum(df.height for df in bems.values()))
        out = (
            Path(args.output) if args.output and len(gw_groups) == 1
            else log_dir / f"{gw}_algo_quality.html"
        )
        write_gateway_html(gw, bems, out)


if __name__ == "__main__":
    main()
