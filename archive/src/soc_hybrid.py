"""
soc_hybrid.py
─────────────
Hybrid SoC algorithm: LUT-based "reset" at high-current events, Coulomb
counting between resets.

State machine
─────────────

             current ≤ -85 A
             OR  100 A ≤ I ≤ 130 A
                    │
    UNINIT ─────────┤ LUT_RESET ────────────────────┐
       │            │     │ condition no longer met  │
       │ (no reset  │     ▼                          │
       │  yet seen) └─ COULOMB                       │
       │                  │ condition met again  ─────┘
       └── output NaN ────┘

UNINIT     — waiting for first high-current window; all outputs are NaN.
LUT_RESET  — per-cell SoC snapped to LUT voltage estimate (high-current IR
             drop pushes voltage off the flat LFP plateau → high accuracy).
COULOMB    — ΔSoC = I_A × Δt_s / (cap_Ah × 3600) × 100 applied identically
             to all cells (series string, same current).  Individual cell SoCs
             diverge/converge only at LUT reset points.

Callable from both the Athena batch pipeline and the MQTT real-time logger:

    # Batch (Athena / log-file CSV) — one call per BEM DataFrame
    df_with_hybrid = apply_hybrid_soc(df, lut_arrays, cfg, capacity_ah=277)

    # Real-time (MQTT) — one tracker per BEM, updated per datagram
    tracker = HybridSoCTracker(cfg, lut_arrays, capacity_ah=277, num_cells=32)
    result  = tracker.update(timestamp, current_a, cells)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import polars as pl

LOG = logging.getLogger(__name__)

# Import the shared LUT lookup from the MQTT logger.
# If soc_hybrid.py lives in a different directory, adjust the import or
# copy estimate_cell_soc / LUT_REFERENCE_CAPACITY here directly.
try:
    from mqtt_bem_soc_logger import estimate_cell_soc, LUT_REFERENCE_CAPACITY
except ImportError:
    # Fallback: copy the constants so the module can be used standalone.
    LUT_REFERENCE_CAPACITY = 277.0

    def estimate_cell_soc(cell_voltage_mv, cell_temp_bucket, current_bucket_scaled,
                          lut, resistance_ohm=0.00025):
        raise ImportError("mqtt_bem_soc_logger not on sys.path — provide estimate_cell_soc")


# ─── Configuration ────────────────────────────────────────────────────────────
@dataclass
class HybridSoCConfig:
    """
    All thresholds for the hybrid algorithm.  Tune per fleet segment.

    Default thresholds are chosen for systems where large C-rate events push
    the cell voltage decisively off the flat LFP OCV plateau:
      - discharge reset : I ≤ -85 A   (~30 % C at 277 Ah)
      - charge reset    : 100 A ≤ I ≤ 130 A  (~36–47 % C at 277 Ah)

    For the gw-000027 data set (max ±45 A) these thresholds are never
    reached — lower them (e.g. -30 A / 35–45 A) for that fleet.
    """
    discharge_reset_a: float = -85.0    # ≤ this → LUT reset
    charge_reset_min_a: float = 100.0   # ┐ in this window
    charge_reset_max_a: float = 130.0   # ┘ → LUT reset
    resistance_ohm: float = 0.00025     # IR compensation
    dt_max_s: float = 300.0             # clamp Δt to avoid huge CC jumps
    cc_clamp: tuple[float, float] = (0.0, 100.0)  # SoC bounds

    def is_reset_point(self, current_a: float) -> bool:
        return (
            current_a <= self.discharge_reset_a
            or self.charge_reset_min_a <= current_a <= self.charge_reset_max_a
        )


# ─── Real-time tracker (one instance per BEM) ────────────────────────────────
@dataclass
class HybridSoCResult:
    soc: list[Optional[float]]   # per-cell SoC; None if UNINIT
    mode: str                    # "UNINIT" | "LUT" | "CC"
    d_soc: float = 0.0           # Δ applied this step (CC mode) or 0.0


class HybridSoCTracker:
    """
    Per-BEM, per-cell hybrid SoC state machine for real-time (MQTT) use.

    Usage
    -----
    tracker = HybridSoCTracker(
        cfg=HybridSoCConfig(),
        lut=make_lut_arrays(df_lut),
        capacity_ah=277.0,
        num_cells=32,
    )

    # Call once per MQTT datagram for this BEM:
    result = tracker.update(
        timestamp=datetime.now(timezone.utc),
        current_a=bemCurrentInst_mA / 1000.0,
        cells=[{"cellVoltage": 3258, "temperature": 22}, ...],  # 32 dicts
    )
    # result.soc[i]  — per-cell hybrid SoC (None until first reset)
    # result.mode    — "LUT" | "CC" | "UNINIT"
    """

    UNINIT = "UNINIT"
    LUT    = "LUT"
    CC     = "CC"

    def __init__(
        self,
        cfg: HybridSoCConfig,
        lut,
        capacity_ah: float,
        num_cells: int = 32,
    ):
        self.cfg = cfg
        self.lut = lut
        self.capacity_ah = capacity_ah
        self.num_cells = num_cells

        self._soc: list[Optional[float]] = [None] * num_cells
        self._last_ts: Optional[datetime] = None
        self._mode = self.UNINIT

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _lut_estimate(self, cell_voltage_mv: float, cell_temp_c: float,
                      current_a: float) -> Optional[float]:
        """Return LUT midpoint SoC for a single cell (None if LUT miss)."""
        current_scaled = current_a / self.capacity_ah * LUT_REFERENCE_CAPACITY
        current_bucket = float(np.floor(current_scaled / 5.0) * 5.0)
        t_bucket = (
            float(np.floor(cell_temp_c / 5.0) * 5.0)
            if cell_temp_c is not None and not np.isnan(cell_temp_c)
            else float("nan")
        )
        s_min, s_max = estimate_cell_soc(
            cell_voltage_mv=float(cell_voltage_mv),
            cell_temp_bucket=t_bucket,
            current_bucket_scaled=current_bucket,
            lut=self.lut,
            resistance_ohm=self.cfg.resistance_ohm,
        )
        if np.isfinite(s_min) and np.isfinite(s_max):
            return (s_min + s_max) / 2.0
        return None

    # ── Public API ────────────────────────────────────────────────────────────
    def update(
        self,
        timestamp: datetime,
        current_a: float,
        cells: list[dict],
    ) -> HybridSoCResult:
        """
        Process one datagram for this BEM.

        Parameters
        ----------
        timestamp : datetime (timezone-aware preferred)
        current_a : pack current in Amperes (negative = discharge)
        cells     : list of per-cell dicts with at least
                    ``cellVoltage`` (mV) and ``temperature`` (°C) keys
        """
        reset = self.cfg.is_reset_point(current_a)

        # ── Coulomb counting delta (used in CC mode) ──────────────────────
        if self._last_ts is not None:
            dt_s = (timestamp - self._last_ts).total_seconds()
            dt_s = max(0.0, min(dt_s, self.cfg.dt_max_s))
        else:
            dt_s = 0.0

        # ΔSoC is the same for all cells (series string, identical current)
        d_soc = (current_a * dt_s / (self.capacity_ah * 3600.0)) * 100.0

        # ── Update per-cell SoC ───────────────────────────────────────────
        new_soc: list[Optional[float]] = []

        if reset:
            mode = self.LUT
            for i in range(self.num_cells):
                if i >= len(cells):
                    new_soc.append(self._soc[i])
                    continue
                c = cells[i]
                v = c.get("cellVoltage")
                t = c.get("temperature")
                if v is not None and v > 0:
                    est = self._lut_estimate(float(v), float(t) if t is not None else float("nan"),
                                             current_a)
                    new_soc.append(est if est is not None else self._soc[i])
                else:
                    new_soc.append(self._soc[i])
        else:
            if all(s is None for s in self._soc):
                mode = self.UNINIT
                new_soc = [None] * self.num_cells
            else:
                mode = self.CC
                lo, hi = self.cfg.cc_clamp
                for s in self._soc:
                    if s is None:
                        new_soc.append(None)
                    else:
                        new_soc.append(float(np.clip(s + d_soc, lo, hi)))

        self._soc = new_soc
        self._last_ts = timestamp
        self._mode = mode
        return HybridSoCResult(soc=new_soc, mode=mode, d_soc=d_soc)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def soc(self) -> list[Optional[float]]:
        return list(self._soc)


# ─── Batch version for Athena / log-file CSV ─────────────────────────────────
def apply_hybrid_soc(
    df: pl.DataFrame,
    lut_arrays,
    cfg: HybridSoCConfig,
    capacity_ah: float,
    num_cells: int = 32,
    timestamp_col: str = "datestamp",
    current_col: str = "current_average",   # Athena: already in A
    current_is_ma: bool = False,            # set True for log-file bemCurrentInst
    volt_col_prefix: str = "cell_voltage_", # Athena: cell_voltage_1..N
    temp_col: str = "cell_temperatures",    # single temp col (Athena) or extTemp (log)
) -> pl.DataFrame:
    """
    Apply the hybrid SoC algorithm to a chronologically sorted per-BEM
    DataFrame.  Works for both:

        • Athena wide output  (current_col="current_average", already A,
                               volt_col_prefix="cell_voltage_",
                               temp_col="cell_temperatures")
        • Log-file CSV        (current_col="bemCurrentInst", current_is_ma=True,
                               volt_col_prefix="cellVoltage_",
                               temp_col="extTemp")

    Adds columns
        soc_hybrid_1 … soc_hybrid_N   (float, NaN until first reset)
        soc_mode                       (string: UNINIT / LUT / CC)
    and returns the augmented DataFrame.
    """
    df = df.sort(timestamp_col)
    n_rows = df.height

    # ── Pre-extract to Python lists for row-by-row iteration ─────────────
    timestamps = df[timestamp_col].to_list()

    raw_curr = df[current_col].to_list()
    if current_is_ma:
        currents_a = [c / 1000.0 if c is not None else 0.0 for c in raw_curr]
    else:
        currents_a = [float(c) if c is not None else 0.0 for c in raw_curr]

    # Temperature — single column (extTemp or cell_temperatures)
    if temp_col in df.columns:
        temps = df[temp_col].to_list()
    else:
        LOG.warning("temp_col '%s' not found; using 20 °C", temp_col)
        temps = [20.0] * n_rows

    # Voltages — dict of col → list
    volt_cols_present = [
        f"{volt_col_prefix}{i}" for i in range(1, num_cells + 1)
        if f"{volt_col_prefix}{i}" in df.columns
    ]
    volt_data = {c: df[c].to_list() for c in volt_cols_present}

    # ── State machine ─────────────────────────────────────────────────────
    tracker = HybridSoCTracker(
        cfg=cfg, lut=lut_arrays, capacity_ah=capacity_ah, num_cells=num_cells,
    )

    hybrid_out: list[list[Optional[float]]] = [[] for _ in range(num_cells)]
    modes: list[str] = []

    for row_idx in range(n_rows):
        ts = timestamps[row_idx]
        # Make timezone-aware if naive
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        current_a = currents_a[row_idx]
        t = temps[row_idx]

        cells = []
        for i in range(1, num_cells + 1):
            col = f"{volt_col_prefix}{i}"
            v = volt_data.get(col, [None] * n_rows)[row_idx]
            cells.append({"cellVoltage": v, "temperature": t})

        result = tracker.update(ts, current_a, cells)
        modes.append(result.mode)
        for i, s in enumerate(result.soc[:num_cells]):
            hybrid_out[i].append(s)

    # ── Append columns to DataFrame ───────────────────────────────────────
    new_cols = [
        pl.Series(f"soc_hybrid_{i+1}", hybrid_out[i], dtype=pl.Float64)
        for i in range(num_cells)
    ]
    new_cols.append(pl.Series("soc_mode", modes, dtype=pl.Utf8))
    return df.with_columns(new_cols)


# ─── Integration notes ────────────────────────────────────────────────────────
"""
──────────────────────────────────────────────────────────────────────────────
A.  Athena / soc_reported_vs_estimate.py
──────────────────────────────────────────────────────────────────────────────
After prepare_base(), before build_wide_result(), insert:

    from soc_hybrid import HybridSoCConfig, apply_hybrid_soc

    cfg = HybridSoCConfig(
        discharge_reset_a  = -85.0,
        charge_reset_min_a = 100.0,
        charge_reset_max_a = 130.0,
    )
    # df_base is long-format; apply per-BEM (here there is only one BEM per run)
    df_base = apply_hybrid_soc(
        df_base,
        lut_arrays   = make_lut_arrays(df_lut),
        cfg          = cfg,
        capacity_ah  = args.capacity,
        num_cells    = args.num_cells,
        timestamp_col   = "datestamp",      # or whatever timestamp col Athena returns
        current_col     = "current_average",
        current_is_ma   = False,
        volt_col_prefix = "cell_voltage_",
        temp_col        = "cell_temperatures",
    )
    # df_base now has soc_hybrid_1..N and soc_mode columns.
    # In build_wide_result() join soc_hybrid_N alongside soc_cell_N.

──────────────────────────────────────────────────────────────────────────────
B.  MQTT real-time  (mqtt_bem_soc_logger.py)
──────────────────────────────────────────────────────────────────────────────
In BemLogger.__post_init__(), add:

    from soc_hybrid import HybridSoCConfig, HybridSoCTracker
    self.hybrid_tracker = HybridSoCTracker(
        cfg          = HybridSoCConfig(),
        lut          = self.lut,
        capacity_ah  = self.capacity_ah,
        num_cells    = self.num_cells,
    )

In BemLogger.on_datagram(), after parsing timestamp and current:

    hybrid_result = self.hybrid_tracker.update(
        timestamp = <parsed datestamp as datetime>,
        current_a = bem_curr_ma / 1000.0,
        cells     = cells,                # already the list-of-dicts from payload
    )
    # hybrid_result.soc[i]  — per-cell; None until first reset
    # hybrid_result.mode    — "LUT" / "CC" / "UNINIT"

Add soc_hybrid_1..N and soc_mode to the log CSV header and row.
──────────────────────────────────────────────────────────────────────────────
"""


# ─── Quick test (run directly against a log-file CSV) ────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Test hybrid SoC on a log-file CSV.")
    p.add_argument("--input",      required=True)
    p.add_argument("--lut-path",   required=True)
    p.add_argument("--capacity",   type=float, default=277.0)
    p.add_argument("--num-cells",  type=int,   default=32)
    p.add_argument("--discharge-reset", type=float, default=-85.0)
    p.add_argument("--charge-min",      type=float, default=100.0)
    p.add_argument("--charge-max",      type=float, default=130.0)
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    # Load LUT
    from mqtt_bem_soc_logger import load_lut, make_lut_arrays
    lut_df = load_lut(args.lut_path)
    lut    = make_lut_arrays(lut_df)

    cfg = HybridSoCConfig(
        discharge_reset_a  = args.discharge_reset,
        charge_reset_min_a = args.charge_min,
        charge_reset_max_a = args.charge_max,
    )

    df = pl.read_csv(args.input, infer_schema_length=2000)
    if df.schema.get("datestamp") != pl.Datetime:
        s = df["datestamp"].cast(pl.Utf8)
        try:
            parsed = s.str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ",
                                        time_zone="UTC", strict=False)
        except Exception:
            parsed = s.str.to_datetime(strict=False)
        df = df.with_columns(datestamp=parsed)

    df_out = apply_hybrid_soc(
        df,
        lut_arrays      = lut,
        cfg             = cfg,
        capacity_ah     = args.capacity,
        num_cells       = args.num_cells,
        timestamp_col   = "datestamp",
        current_col     = "bemCurrentInst",
        current_is_ma   = True,
        volt_col_prefix = "cellVoltage_",
        temp_col        = "extTemp",
    )

    mode_counts = df_out["soc_mode"].value_counts().sort("soc_mode")
    print("Mode counts:")
    for row in mode_counts.iter_rows(named=True):
        print(f"  {row['soc_mode']:8s}: {row['count']}")

    hybrid_cols = [c for c in df_out.columns if c.startswith("soc_hybrid_")]
    valid = df_out.filter(pl.col("soc_hybrid_1").is_not_null())
    if valid.height > 0:
        print(f"\nsoc_hybrid range (cell 1): "
              f"{valid['soc_hybrid_1'].min():.2f} → {valid['soc_hybrid_1'].max():.2f}")
        print(f"  vs reported soc:          "
              f"{valid['soc'].min()} → {valid['soc'].max()}")
    else:
        print("\nNo LUT resets triggered — lower --discharge-reset and/or --charge-min thresholds.")
        print(f"Current range in file: "
              f"{df_out['bemCurrentInst'].min()/1000:.1f} A → "
              f"{df_out['bemCurrentInst'].max()/1000:.1f} A")

    out = args.output or args.input.replace(".log", "_hybrid.csv")
    df_out.write_csv(out)
    print(f"\nWrote {out}")
