"""
mqtt_bem_pack_soc_logger.py
───────────────────────────
Listen to MQTT ``getMeasuredVI`` replies from one or more gateways and run a
**pack-level** SoC estimation against the same LUT used by
``mqtt_bem_soc_logger.py``, writing one row per message to a per-BEM,
daily-rotated CSV.

This is the pack-level companion to ``mqtt_bem_soc_logger.py``:
  • ``getLatestDatagram`` (per-cell) arrives slowly → per-cell logger.
  • ``getMeasuredVI``     (pack-level) arrives more often → this script.

Topic
-----
    ``{gateway}/local/bem/response/{bem_id}/getMeasuredVI``

Sample payload
--------------
    {
      "type": "getMeasuredVI", "sourceAddress": 1,
      "voltage": 105301,          # pack voltage in mV (sum of cells)
      "current": -348,            # pack current in mA
      "battery_SoC": 56,          # reported SoC
      "external_temp": 19,        # °C
      "timestamp": 63210094, ...
    }

Algorithm
---------
1. Derive pack-cell voltage:   ``V_cell_mV = voltage / num_cells``
2. Convert current to A:       ``I_A = current / 1000``
3. Scale to LUT capacity:      ``I' = I_A / capacity_ah * 277``
   then bucket to nearest 5 A: ``floor(I'/5)*5``
4. Bucket ``external_temp`` to nearest 5 °C.
5. Look up nearest LUT (T, I) bin, IR-compensate V, search V→DoD → (soc_min, soc_max).
6. Append midpoint to a 13-sample rolling deque; ``soc_rolling`` = mean once ≥12 samples.

CSV columns
-----------
    datestamp, timestamp, battery_SoC,
    current_mA, voltage_mV, pack_cell_voltage_mV, external_temp,
    soc_min, soc_max, soc_rolling

Usage
-----
    uv run mqtt_bem_pack_soc_logger.py \\
        --gateway gw-000105 --gateway gw-000106 \\
        --num-bems 4 \\
        --broker mqtt.example.com --port 1883 \\
        --capacity 277 --num-cells 32 \\
        --lut-path data/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \\
        --log-dir ./logs

Offline sanity check against the pasted sample payload::

    uv run mqtt_bem_pack_soc_logger.py \\
        --gateway gw-000105 --num-bems 4 --broker dummy \\
        --capacity 277 --num-cells 32 \\
        --lut-path /path/to/lut.parquet \\
        --simulate-payload sample_vi.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import logging.handlers
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiomqtt
import numpy as np
import polars as pl


# ─── Constants ───────────────────────────────────────────────────────────────
WINDOW_SIZE_SOC = 13                       # same as soc_reported_vs_estimate.py
MIN_SAMPLES_ROLLING = WINDOW_SIZE_SOC - 1  # min_samples = window-1
LUT_REFERENCE_CAPACITY = 277.0
RESISTANCE_OHM = 0.00025
LUT_SOH_MIN = 98
LUT_SOH_MAX = 102

LOG = logging.getLogger("mqtt_bem_pack")


# ─── LUT loading (identical to per-cell version) ─────────────────────────────
def load_lut(lut_path: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(lut_path)
        .with_columns((pl.col("q_removed_ah") / 277 * 100).round(0).alias("dod"))
        .filter(pl.col("dod").is_between(0, 100, closed="both"))
        .group_by(["soh", "cell_temperatures", "current_average", "dod"])
        .agg(pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean"))
        .sort(["soh", "cell_temperatures", "current_average", "dod"])
        .filter(pl.col("soh").is_between(LUT_SOH_MIN, LUT_SOH_MAX, closed="both"))
        .group_by(["cell_temperatures", "current_average", "dod"])
        .agg(pl.col("cell_voltages_mean").mean().round(0).alias("cell_voltages_mean"))
        .with_columns(
            pl.col("dod").count()
            .over(["cell_temperatures", "current_average"])
            .alias("dod_count")
        )
        .filter(pl.col("dod_count") >= 90)
        .drop("dod_count")
        .sort(["cell_temperatures", "current_average", "dod"])
        .collect()
    )


@dataclass
class LutArrays:
    """Pre-extracted numpy arrays / per-bin caches for fast per-row lookup."""
    unique_temps: np.ndarray
    available_currents: dict           # {temp: np.ndarray of currents}
    bin_curves: dict                   # {(temp, curr): (dod_sorted, volt_sorted)}


def make_lut_arrays(lut: pl.DataFrame) -> LutArrays:
    temp = lut["cell_temperatures"].to_numpy().astype(float)
    curr = lut["current_average"].to_numpy().astype(float)
    volt = lut["cell_voltages_mean"].to_numpy().astype(float)
    dod = lut["dod"].to_numpy().astype(float)

    unique_temps = np.unique(temp)
    available_currents = {
        float(t): np.unique(curr[temp == t]) for t in unique_temps
    }

    bin_curves: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for t in unique_temps:
        for c in available_currents[float(t)]:
            mask = (temp == t) & (curr == c)
            bd, bv = dod[mask], volt[mask]
            valid = np.isfinite(bv) & np.isfinite(bd) & (bv >= 0)
            bd, bv = bd[valid], bv[valid]
            if bd.size == 0:
                continue
            idx = np.argsort(bd)
            bin_curves[(float(t), float(c))] = (bd[idx], bv[idx])

    return LutArrays(
        unique_temps=unique_temps,
        available_currents=available_currents,
        bin_curves=bin_curves,
    )


def estimate_soc(
    cell_voltage_mv: float,
    cell_temp_bucket: float,
    current_bucket_scaled: float,
    lut: LutArrays,
    resistance_ohm: float = RESISTANCE_OHM,
) -> tuple[float, float]:
    """Return (soc_min, soc_max) for a single (V_cell_mV, T_°C, I_A_scaled) sample.

    Identical algorithm to the per-cell version — the only difference is that
    here ``cell_voltage_mv`` is derived as ``pack_voltage / num_cells``.
    """
    if cell_temp_bucket is None or np.isnan(cell_temp_bucket):
        matched_temp = float(np.median(lut.unique_temps))
    else:
        matched_temp = float(
            lut.unique_temps[np.argmin(np.abs(lut.unique_temps - cell_temp_bucket))]
        )

    avail = lut.available_currents.get(matched_temp)
    if avail is None or avail.size == 0:
        return (float("nan"), float("nan"))

    if current_bucket_scaled is None or np.isnan(current_bucket_scaled):
        matched_curr = float(avail[np.argmin(np.abs(avail))])
    else:
        matched_curr = float(
            avail[np.argmin(np.abs(avail - current_bucket_scaled))]
        )

    # IR compensation from actual to LUT current
    r_mohm = resistance_ohm * 1000.0
    v_adj = cell_voltage_mv - r_mohm * (current_bucket_scaled - matched_curr)

    curve = lut.bin_curves.get((matched_temp, matched_curr))
    if curve is None:
        return (float("nan"), float("nan"))
    bin_dod, bin_volt = curve

    v_high, v_low = bin_volt[0], bin_volt[-1]

    if v_adj > v_high:
        dod_lo = dod_hi = 0.0
    elif v_adj < v_low:
        dod_lo = dod_hi = 100.0
    else:
        flipped_v = bin_volt[::-1]
        flipped_d = bin_dod[::-1]
        idx_right = int(np.clip(
            np.searchsorted(flipped_v, v_adj, side="left"),
            0, len(flipped_v) - 1,
        ))
        idx_left = int(np.clip(idx_right - 1, 0, len(flipped_v) - 1))
        d1, d2 = float(flipped_d[idx_left]), float(flipped_d[idx_right])
        dod_lo = max(min(d1, d2), 0.0)
        dod_hi = min(max(d1, d2), 100.0)

    soc_min = max(0.0, min(100.0, 100.0 - dod_hi))
    soc_max = max(0.0, min(100.0, 100.0 - dod_lo))
    return soc_min, soc_max


# ─── CSV-aware daily-rotating handler ────────────────────────────────────────
class CsvDailyRotatingHandler(logging.handlers.TimedRotatingFileHandler):
    """Daily rotation (UTC midnight) that re-writes the CSV header to every
    new file so each rotated file is independently parseable."""

    def __init__(self, filename: str, header: str, backup_count: int = 30):
        self._header = header
        super().__init__(
            filename,
            when="midnight",
            utc=True,
            backupCount=backup_count,
        )
        self._maybe_write_header()

    def _maybe_write_header(self) -> None:
        if self.stream is None:
            self.stream = self._open()
        if self.stream.tell() == 0:
            self.stream.write(self._header + "\n")
            self.stream.flush()

    def doRollover(self) -> None:                         # noqa: N802 (stdlib)
        super().doRollover()
        self._maybe_write_header()


# ─── Per-BEM pack-level state ────────────────────────────────────────────────
@dataclass
class BemPackLogger:
    gateway: str
    bem_id: int
    num_cells: int
    capacity_ah: float
    lut: LutArrays
    log_dir: Path
    bem_model: str = "P34"
    serial: str = "UNKNOWN"
    rolling: deque = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.rolling = deque(maxlen=WINDOW_SIZE_SOC)

        header = ",".join([
            "datestamp", "timestamp", "battery_SoC",
            "current_mA", "voltage_mV", "pack_cell_voltage_mV", "external_temp",
            "soc_min", "soc_max", "soc_rolling",
        ])

        self.log_dir.mkdir(parents=True, exist_ok=True)
        fname = self.log_dir / (
            f"{self.gateway}-{self.serial}-{self.bem_model}-"
            f"{self.bem_id:02d}-pack.log"
        )

        name = f"bem_pack.{self.gateway}.{self.bem_id}"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False        # keep CSV data out of the app log
        if not self.logger.handlers:
            h = CsvDailyRotatingHandler(str(fname), header=header)
            h.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(h)

        LOG.info("[%s/bem %d] pack-logging to %s",
                 self.gateway, self.bem_id, fname)

    # -----------------------------------------------------------------
    def on_measured_vi(self, payload: dict) -> None:
        # getMeasuredVI replies use these field names (per the sample payload)
        voltage_mv = payload.get("voltage")
        current_ma = payload.get("current")
        soc = payload.get("battery_SoC")
        ext_temp = payload.get("external_temp")
        ts = payload.get("timestamp")

        if voltage_mv is None or current_ma is None:
            LOG.warning("[%s/bem %d] malformed getMeasuredVI (V=%s, I=%s), skipping",
                        self.gateway, self.bem_id, voltage_mv, current_ma)
            return

        # Wall-clock for the CSV. The `timestamp` field is device-side and not
        # a real epoch, so we stamp arrivals with UTC now() (ISO-8601).
        datestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

        # Derive per-cell voltage from pack voltage (already in mV)
        pack_cell_voltage_mv = float(voltage_mv) / float(self.num_cells)

        # Scale current to LUT reference capacity, then bucket to nearest 5 A
        current_a = float(current_ma) / 1000.0
        current_scaled = current_a / self.capacity_ah * LUT_REFERENCE_CAPACITY
        current_bucket = float(np.floor(current_scaled / 5.0) * 5.0)

        # Bucket external temperature
        if ext_temp is None:
            t_bucket = float("nan")
        else:
            t_bucket = float(np.floor(float(ext_temp) / 5.0) * 5.0)

        soc_min, soc_max = estimate_soc(
            cell_voltage_mv=pack_cell_voltage_mv,
            cell_temp_bucket=t_bucket,
            current_bucket_scaled=current_bucket,
            lut=self.lut,
        )

        if np.isfinite(soc_min) and np.isfinite(soc_max):
            self.rolling.append((soc_min + soc_max) / 2.0)

        if len(self.rolling) >= MIN_SAMPLES_ROLLING:
            soc_rolling_str = f"{float(np.mean(self.rolling)):.3f}"
        else:
            soc_rolling_str = ""

        soc_min_str = f"{soc_min:.3f}" if np.isfinite(soc_min) else ""
        soc_max_str = f"{soc_max:.3f}" if np.isfinite(soc_max) else ""

        row = [
            datestamp,
            str(ts) if ts is not None else "",
            str(soc) if soc is not None else "",
            str(current_ma),
            str(voltage_mv),
            f"{pack_cell_voltage_mv:.1f}",
            str(ext_temp) if ext_temp is not None else "",
            soc_min_str, soc_max_str, soc_rolling_str,
        ]
        self.logger.info(",".join(row))


# ─── MQTT runner ─────────────────────────────────────────────────────────────
async def gateway_subscriber(
    *,
    gateway: str,
    num_bems: int,
    broker: str,
    port: int,
    username: Optional[str],
    password: Optional[str],
    capacity_ah: float,
    num_cells: int,
    lut: LutArrays,
    log_dir: Path,
    serial_map: dict[int, str],
    bem_model: str,
    stop: asyncio.Event,
) -> None:
    bems: dict[int, BemPackLogger] = {
        bem_id: BemPackLogger(
            gateway=gateway,
            bem_id=bem_id,
            num_cells=num_cells,
            capacity_ah=capacity_ah,
            lut=lut,
            log_dir=log_dir,
            bem_model=bem_model,
            serial=serial_map.get(bem_id, f"BEM{bem_id:02d}"),
        )
        for bem_id in range(1, num_bems + 1)
    }

    topic = "local/bem/response/+/getMeasuredVI"
    LOG.info("[%s] subscribing to %s (BEMs 1..%d)", gateway, topic, num_bems)

    backoff = 1.0
    while not stop.is_set():
        try:
            async with aiomqtt.Client(
                hostname=broker,
                port=port,
                username=username,
                password=password,
                identifier=f"bem-pack-soc-logger-{gateway}",
            ) as client:
                await client.subscribe(topic, qos=1)
                LOG.info("[%s] connected to %s:%d", gateway, broker, port)
                backoff = 1.0
                async for msg in client.messages:
                    try:
                        parts = str(msg.topic).split("/")
                        # .../bem/response/{bem_id}/getMeasuredVI
                        bem_id = int(parts[-2])
                    except (ValueError, IndexError):
                        LOG.warning("[%s] unexpected topic: %s", gateway, msg.topic)
                        continue
                    if bem_id not in bems:
                        continue
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except Exception:
                        LOG.exception("[%s/bem %d] payload parse failed",
                                      gateway, bem_id)
                        continue
                    # Guard against unrelated payloads landing on this topic
                    if payload.get("type") and payload["type"] != "getMeasuredVI":
                        continue
                    try:
                        bems[bem_id].on_measured_vi(payload)
                    except Exception:
                        LOG.exception("[%s/bem %d] getMeasuredVI processing failed",
                                      gateway, bem_id)
        except aiomqtt.MqttError as e:
            if stop.is_set():
                break
            LOG.warning("[%s] MQTT error: %s — reconnecting in %.1fs",
                        gateway, e, backoff)
            try:
                await asyncio.wait_for(stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2.0, 60.0)


# ─── Setup helpers ───────────────────────────────────────────────────────────
def setup_script_logging(log_dir: Path) -> None:
    """Per user preference: every script writes to a (rotating) log file by default."""
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    file_h = logging.handlers.RotatingFileHandler(
        log_dir / "mqtt_bem_pack_soc_logger.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(fmt)
    root.addHandler(stream_h)


def parse_serial_map(path: str | None) -> dict[str, dict[int, str]]:
    """Optional JSON: ``{"gw-000105": {"1": "BE20000157", "2": "BE20000158"}}``."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text())
    return {gw: {int(k): v for k, v in m.items()} for gw, m in raw.items()}


# ─── Simulation path (offline sanity check) ──────────────────────────────────
def run_simulate(args: argparse.Namespace, lut: LutArrays) -> None:
    """Process a single JSON payload from disk and exit — no MQTT involved."""
    payload = json.loads(Path(args.simulate_payload).read_text())
    bem_id = int(payload.get("sourceAddress", 1))
    gateway = args.gateway[0]
    serial_map = parse_serial_map(args.serial_map).get(gateway, {})

    bem = BemPackLogger(
        gateway=gateway,
        bem_id=bem_id,
        num_cells=args.num_cells,
        capacity_ah=args.capacity,
        lut=lut,
        log_dir=Path(args.log_dir),
        bem_model=args.bem_model,
        serial=serial_map.get(bem_id, f"BEM{bem_id:02d}"),
    )
    bem.on_measured_vi(payload)
    LOG.info("Simulated getMeasuredVI written for %s / bem %d", gateway, bem_id)


# ─── Main ────────────────────────────────────────────────────────────────────
async def main_async(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    setup_script_logging(log_dir)

    LOG.info("Loading LUT from %s …", args.lut_path)
    lut_df = load_lut(args.lut_path)
    LOG.info("LUT loaded: %d rows", lut_df.height)
    lut = make_lut_arrays(lut_df)

    if args.simulate_payload:
        run_simulate(args, lut)
        return

    serial_map = parse_serial_map(args.serial_map)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    tasks = [
        asyncio.create_task(
            gateway_subscriber(
                gateway=gw,
                num_bems=args.num_bems,
                broker=args.broker,
                port=args.port,
                username=args.username,
                password=args.password,
                capacity_ah=args.capacity,
                num_cells=args.num_cells,
                lut=lut,
                log_dir=log_dir,
                serial_map=serial_map.get(gw, {}),
                bem_model=args.bem_model,
                stop=stop,
            ),
            name=f"sub-{gw}",
        )
        for gw in args.gateway
    ]
    LOG.info("Started %d gateway listener task(s)", len(tasks))

    await stop.wait()
    LOG.info("Shutting down …")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    LOG.info("Done.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MQTT BEM pack-level SoC logger (getMeasuredVI, daily-rotated CSV).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gateway", action="append", required=True,
                   help="Gateway name (repeat for multiple, e.g. gw-000105)")
    p.add_argument("--num-bems", type=int, required=True,
                   help="BEM bus addresses 1..N to subscribe to")
    p.add_argument("--broker", required=True, help="MQTT broker hostname")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--capacity", type=float, required=True,
                   help="Battery capacity in Ah")
    p.add_argument("--num-cells", type=int, default=32,
                   help="Cells in series per pack (default: 32)")
    p.add_argument("--lut-path", required=True, help="Path to LUT parquet file")
    p.add_argument("--log-dir", default="./logs",
                   help="Directory for per-BEM .log files and the script log")
    p.add_argument("--bem-model", default="P34",
                   help="Model token in the filename (default: P34)")
    p.add_argument("--serial-map", default=None,
                   help="Optional JSON: {gateway: {bem_id: serial}}")
    p.add_argument("--simulate-payload", default=None,
                   help="Path to a JSON getMeasuredVI payload; process once and exit.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
