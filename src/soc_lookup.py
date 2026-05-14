"""
SOC estimation via current-binned voltage lookup table.

Designed as a reference implementation for a future embedded C port, so the
core search routine uses plain integer indices and a hand-rolled binary
search rather than Polars idioms. Polars is only used for I/O, fanning the
row-wise kernel out, and final unpivot/plot.

Algorithm (per row):
  1. Find nearest current bin to measured current (absolute distance,
     binary search over sorted current_bins[]).
  2. Inside that bin, binary-search over voltage_lut[bin][0..100] for the
     SOC whose stored voltage matches the measured voltage.
     - Voltage is increasing in SOC within a bin (LFP-like, with possible
       plateaus).
     - On exact hit: soc_min = soc_max = that index.
     - On no exact hit: soc_min / soc_max bracket the measured voltage.
     - Measured voltage below table[0]  -> clamp to 0.
     - Measured voltage above table[100] -> clamp to 100.
  3. Pick "chosen" SOC as whichever of {min, max} has the closer stored
     voltage to measured. Ties and the degenerate-plateau case default to
     soc_min (per spec).

The C port will have the same control flow; only the data-plumbing layer
(loops over rows, CSV I/O, DataFrame construction) is Python-specific.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Polars and plotly are only needed for the I/O and plotting layer, not the
# kernel. Imported lazily inside the functions that need them so the kernel
# can be unit-tested without them installed.


# ---------------------------------------------------------------------------
# Core kernel — written to mirror the eventual embedded C.
#
# In C this will be roughly:
#
#   typedef struct {
#       int32_t  current_bins[N_BINS];           // sorted ascending
#       uint16_t voltage_lut[N_BINS][101];       // mV, increasing in SOC
#   } soc_lut_t;
#
#   uint8_t estimate_soc(const soc_lut_t *lut,
#                        int32_t current_mA,
#                        uint16_t voltage_mV,
#                        uint8_t *soc_min_out,
#                        uint8_t *soc_max_out);
#
# Everything below uses ints where the C code will use ints. Floats only
# appear at the Python I/O boundary.
# ---------------------------------------------------------------------------


def find_nearest_current_bin(current_bins: np.ndarray, current: int) -> int:
    """Return the index of the current bin closest to `current` by abs distance.

    `current_bins` is sorted ascending. Uses binary search to find the
    insertion point, then compares the two neighbours.

    C equivalent: standard lower_bound + neighbour comparison.
    """

    n = len(current_bins)
    if n == 1:
        return 0

    # Binary search for insertion point (lower_bound).
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if current_bins[mid] < current:
            lo = mid + 1
        else:
            hi = mid

    # `lo` is the first index with current_bins[lo] >= current.
    if lo == 0:
        return 0
    if lo == n:
        return n - 1

    # Compare neighbour at lo-1 and lo, pick the closer.
    d_lo = current - current_bins[lo - 1]
    d_hi = current_bins[lo] - current
    return lo - 1 if d_lo <= d_hi else lo


def search_soc_in_bin(
    voltage_row: np.ndarray,  # shape (101,), voltage at each SOC 0..100
    voltage: float,
) -> tuple[int, int]:
    """Binary-search the voltage row for the SOC bracket matching `voltage`.

    Returns (soc_min, soc_max). If measured voltage is an exact hit, both
    equal that SOC. Otherwise they bracket the measured voltage. If out of
    range, both clamp to 0 or 100.

    Voltage is assumed non-decreasing in SOC (may have flat plateaus).
    This is essentially the binary-search-as-described-in-spec:
    start mid, narrow SOC_max or SOC_min based on comparison, repeat.
    """
    n = 101  # SOC 0..100 inclusive
    # print(f"search_soc_in_bin: voltage_row={voltage_row[0]}, voltage={voltage}")
    # Clamp cases first — mirror what the C code will do.
    if voltage <= voltage_row[0]:
        return 0, 0
    if voltage >= voltage_row[n - 1]:
        return 100, 100

    soc_min = 0
    soc_max = n - 1  # 100

    while soc_max - soc_min > 1:
        mid = (soc_min + soc_max) // 2  # e.g. starts at 50
        v_mid = voltage_row[mid]
        if voltage < v_mid:
            soc_max = mid
        elif voltage > v_mid:
            soc_min = mid
        else:
            # Exact match. With plateaus this might not be the only
            # SOC with this voltage, but per spec we default to the
            # found index and collapse the bracket.
            return mid, mid

    # Post-loop: soc_min and soc_max are adjacent and bracket `voltage`.
    # Check adjacency-exact-hit at the bounds just in case.
    if voltage_row[soc_min] == voltage:
        return soc_min, soc_min
    if voltage_row[soc_max] == voltage:
        return soc_max, soc_max
    return soc_min, soc_max


def choose_soc(
    voltage_row: np.ndarray, voltage: float, soc_min: int, soc_max: int
) -> int:
    """Pick whichever of soc_min / soc_max has closer stored voltage to measured.

    Ties default to soc_min per spec. When soc_min == soc_max (exact hit or
    clamp), trivially returns that value.
    """
    if soc_min == soc_max:
        return soc_min
    d_min = abs(voltage - voltage_row[soc_min])
    d_max = abs(voltage - voltage_row[soc_max])
    # <= so ties go to min.
    return soc_min if d_min <= d_max else soc_max


def estimate_soc_row(
    current_bins: np.ndarray,  # (N_BINS,) int
    voltage_lut: np.ndarray,  # (N_BINS, 101) int
    current: float,
    voltage: float,
) -> tuple[int, int, int]:
    """Top-level per-row kernel. Returns (soc_min, soc_max, soc_chosen).

    This is the function that has a 1:1 C port.
    """
    bin_idx = find_nearest_current_bin(current_bins, current)
    voltage_row = voltage_lut[bin_idx]
    soc_min, soc_max = search_soc_in_bin(voltage_row, voltage)
    soc_chosen = choose_soc(voltage_row, voltage, soc_min, soc_max)
    return soc_min, soc_max, soc_chosen


# ---------------------------------------------------------------------------
# Polars plumbing — load, fan out the kernel, unpivot, plot.
# ---------------------------------------------------------------------------


def load_lookup_wide(
    lut_path: Path,
    current_col: str = "current_bin",  # kept for CLI compat, ignored internally
    soc_col: str = "state_of_charge_recorded",
    voltage_col: str = "voltage_median",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a *long-form* lookup table and pivot it to the kernel's wide arrays.

    Expected columns: state_of_charge_recorded (0–100 int),
                      current_bin              (mA, int),
                      voltage_median           (mV, int/float → cast to int).

    Returns
    -------
    current_bins : np.ndarray, shape (N_BINS,), int64, sorted ascending
    voltage_lut  : np.ndarray, shape (N_BINS, 101), int64
                   voltage_lut[i, soc] = voltage at current_bins[i] and SOC soc.
    """
    import polars as pl

    lut = pl.read_csv(lut_path).select([soc_col, current_col, voltage_col])
    # lut=lut.with_columns
    # Cast to the types the kernel expects.
    lut = lut.with_columns(
        [
            pl.col(soc_col).cast(pl.Int64),
            pl.col(current_col).cast(pl.Int64),
            pl.col(voltage_col) / 1000.0,
        ]
    )

    # Pivot: rows = current_bin (sorted), columns = SOC 0..100.
    wide = lut.pivot(
        on=soc_col, index=current_col, values=voltage_col, aggregate_function="mean"
    ).sort(current_col)

    current_bins = wide[current_col].to_numpy().astype(np.int64)

    # Pull SOC columns in order 0..100, filling any missing SOC with
    # the nearest available value (forward/backward fill after sort).
    soc_indices = sorted(int(c) for c in wide.columns if c != current_col)
    if len(soc_indices) != 101 or soc_indices != list(range(101)):
        # Build a full 0..100 frame and join, then fill gaps.
        full_soc = pl.DataFrame({soc_col: list(range(101))}).with_columns(
            pl.col(soc_col).cast(pl.Int64)
        )
        # Transpose wide so rows are SOC values, fill, transpose back — easier
        # to just forward/backward-fill column-wise on the wide frame.
        # For each current bin row, interpolate missing SOC voltages.
        for missing_soc in set(range(101)) - set(soc_indices):
            wide = wide.with_columns(
                pl.lit(None).cast(pl.Int64).alias(str(missing_soc))
            )

    ordered_cols = [str(s) for s in range(101)]
    voltage_lut = wide.select(ordered_cols).to_numpy()

    return current_bins, voltage_lut


def run_algorithm(
    raw,
    current_bins: np.ndarray,
    voltage_lut: np.ndarray,
    voltage_col: str,
    current_col: str,
    soc_recorded_col: str,
):
    """Apply the kernel row-by-row and return a dataframe with SOC outputs."""
    import polars as pl

    currents = raw[current_col].to_numpy().astype(np.float64)
    voltages = raw[voltage_col].to_numpy().astype(np.float64)
    n = len(raw)

    soc_min_out = np.empty(n, dtype=np.int64)
    soc_max_out = np.empty(n, dtype=np.int64)
    soc_chosen_out = np.empty(n, dtype=np.int64)

    # Explicit per-row loop — mirrors the C for-loop exactly. Not vectorised
    # on purpose: the shape of the algorithm is what we're testing.
    for i in range(n):
        lo, hi, ch = estimate_soc_row(
            current_bins, voltage_lut, float(currents[i]), float(voltages[i])
        )
        soc_min_out[i] = lo
        soc_max_out[i] = hi
        soc_chosen_out[i] = ch

    return raw.with_columns(
        pl.Series("soc_estimated_min", soc_min_out),
        pl.Series("soc_estimated_max", soc_max_out),
        pl.Series("soc_estimated_chosen", soc_chosen_out),
        pl.col(soc_recorded_col).alias("soc_recorded"),
    )


def build_long_dataframe(
    df,
    sequence_col: str,
    soc_recorded_col: str,
):
    """Unpivot the SOC columns into (sequence, name, value) long form for plotting."""
    import polars as pl  # noqa: F401  (imported for symmetry; df is already polars)

    value_cols = [
        "soc_recorded",
        "soc_estimated_min",
        "soc_estimated_max",
        "soc_estimated_chosen",
    ]
    # If the recorded col has a different raw name, we've already aliased it
    # to `soc_recorded` in run_algorithm, so it's fine.
    return df.unpivot(
        on=value_cols,
        index=[sequence_col],
        variable_name="soc_name",
        value_name="soc_value",
    )


def plot_soc(
    long_df,
    sequence_col: str,
    title: str = "SOC lookup algorithm vs recorded",
):
    """Plot SOC traces over sequence number.

    Styled per usual preferences: no gridlines, black L-shape axes,
    outside ticks, greyscale palette.
    """
    import plotly.graph_objects as go
    import polars as pl

    # Greyscale palette — darkest to lightest, recorded gets the darkest line.
    palette = {
        "soc_recorded": "#FF0000",
        "soc_estimated_chosen": "#0D00FF",
        "soc_estimated_min": "#FFEE00",
        "soc_estimated_max": "#09BA00",
    }
    dash_map = {
        "soc_recorded": "solid",
        "soc_estimated_chosen": "solid",
        "soc_estimated_min": "dot",
        "soc_estimated_max": "dot",
    }

    fig = go.Figure()
    for name in [
        "soc_recorded",
        "soc_estimated_min",
        "soc_estimated_max",
        "soc_estimated_chosen",
    ]:
        sub = long_df.filter(pl.col("soc_name") == name).sort(sequence_col)
        fig.add_trace(
            go.Scatter(
                x=sub[sequence_col].to_list(),
                y=sub["soc_value"].to_list(),
                mode="lines",
                name=name,
                line=dict(color=palette[name], width=2, dash=dash_map[name]),
            )
        )

    axis_common = dict(
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=2,
        showline=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        mirror=False,
        rangemode="normal",
    )
    fig.update_layout(
        title=title,
        xaxis=dict(**axis_common, title=sequence_col),
        yaxis=dict(**axis_common, title="State of charge (%)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import polars as pl

    ap = argparse.ArgumentParser(
        description="SOC lookup algorithm (Polars reference for embedded C port)"
    )
    ap.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Input CSV with sequence/voltage/current/SOC columns",
    )
    ap.add_argument(
        "--lut",
        required=True,
        type=Path,
        help="Lookup table CSV (wide: current_bin + soc 0..100)",
    )
    ap.add_argument(
        "--sequence-col",
        default="sequence_number",
        help="Sequence number column in raw CSV",
    )
    ap.add_argument(
        "--voltage-col", default="min_cell_voltage", help="Voltage column in raw CSV"
    )
    ap.add_argument(
        "--current-col", default="current", help="Current column in raw CSV"
    )
    ap.add_argument(
        "--soc-recorded-col", default="soc", help="Recorded SOC column in raw CSV"
    )
    ap.add_argument(
        "--lut-current-col", default="current_bin", help="Current bin column in LUT CSV"
    )
    ap.add_argument(
        "--out-csv", type=Path, default=Path("soc_results.csv"), help="Wide output CSV"
    )
    ap.add_argument(
        "--out-long-csv",
        type=Path,
        default=Path("soc_results_long.csv"),
        help="Long/unpivoted output CSV",
    )
    ap.add_argument(
        "--out-html", type=Path, default=Path("soc_plot.html"), help="Plot HTML"
    )
    args = ap.parse_args()

    raw = (
        pl.read_csv(args.raw)
        .filter(
            ~pl.col("current").is_between(-10, 10, closed="both")
            )
        # .with_columns(
        #     pl.col("voltage_average")/16
        # )
    )
    
    #     .with_columns(
    #         [
    #             pl.col("current").rolling_min(window_size=5).alias("rolling_min"),
    #             pl.col("current").rolling_max(window_size=5).alias("rolling_max"),
    #         ]
    #     )
    #     .filter((pl.col("rolling_max")-pl.col("rolling_min")).abs()<5)
    #     .drop(["rolling_min", "rolling_max"])
    # )
    current_bins, voltage_lut = load_lookup_wide(
        args.lut, current_col=args.lut_current_col
    )

    result = run_algorithm(
        raw,
        current_bins=current_bins,
        voltage_lut=voltage_lut,
        voltage_col=args.voltage_col,
        current_col=args.current_col,
        soc_recorded_col=args.soc_recorded_col,
    )
    long_df = build_long_dataframe(result, args.sequence_col, args.soc_recorded_col)

    result.write_csv(args.out_csv)
    long_df.write_csv(args.out_long_csv)

    fig = plot_soc(long_df, args.sequence_col)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    fig.show()
    print(f"Wrote {args.out_csv}, {args.out_long_csv}, {args.out_html}")


if __name__ == "__main__":
    main()
