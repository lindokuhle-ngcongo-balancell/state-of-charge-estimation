# soc-estimation

Battery State-of-Charge (SoC) estimation against a parquet OCV lookup table
(temperature × current × DoD), with IR compensation and a per-cell rolling
window. The **same algorithm and code path** drives all three data sources:
CSV files, AWS Athena, and real-time MQTT streams.

[![CI](https://github.com/balancell/state-of-charge-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/balancell/state-of-charge-estimation/actions/workflows/ci.yml)

---

## How it works

The estimator processes battery telemetry one tick at a time and produces a
per-cell SoC bracket plus a rolling-mean SoC for every cell in the pack.

**Per-tick per-cell algorithm:**

1. **Scale & bucket current** — `I' = I_A / capacity_ah × 277`, then floor to
   the nearest 5 A (matches the LUT's `current_average` grid).
2. **Bucket temperature** — floor cell temperature to the nearest 5 °C.
3. **Snap to LUT bin** — find the nearest available `(temperature, current)`
   bin in the parquet LUT.
4. **IR-compensate the cell voltage**
   `V_adj_mV = V_meas_mV − R_mΩ · (I_actual − I_lut_bin)`  (R = 0.25 mΩ default).
5. **Interpolate** — search the bin's V→DoD curve to recover a
   `(soc_min, soc_max)` bracket.
6. **Rolling mean** — push `(soc_min + soc_max)/2` into a 13-sample per-cell
   deque; `soc_rolling` is reported once at least 12 samples are present.

The CSV / Athena / MQTT entry points all funnel into the same shared functions
in `soc_estimation/bem_estimator.py` and write the same output schema — only the
input adapter differs.

---

## Prerequisites

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) **or** `pip`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/balancell/state-of-charge-estimation.git
cd state-of-charge-estimation

# Install everything (project + dev tools) into a managed .venv
uv sync --extra dev

# Or with plain pip into your own venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

> **Running commands.** The examples below use `uv run <cmd>`, which uses the
> project's `.venv` automatically — no activation needed. If you prefer the
> bare command (`pytest`, `soc-estimate-csv`, `ruff`, …), activate the venv
> first: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).

---

## Configuration

### AWS Athena users

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```dotenv
# .env
AWS_PROFILE=infra_dev
AWS_REGION=eu-west-1
ATHENA_DATABASE=battery_data
ATHENA_TABLE=measurements
ATHENA_S3_OUTPUT=s3://your-bucket/athena-results/
```

The `.env` file is loaded automatically at startup. All values can be overridden by the equivalent CLI flag (e.g. `--database`).

> **Never commit `.env` to source control** — it is listed in `.gitignore`.

---

## OCV Lookup Table

All three CLIs consume the same parquet LUT (a 4-D table of OCV vs DoD across
temperature, current, and SoH). At load time the LUT is filtered to a healthy
SoH band (98–102 %), averaged across SoH, and reshaped into one DoD→voltage
curve per `(temperature, current)` bin.

| Column            | Type   | Description |
|-------------------|--------|-------------|
| `soh`             | int    | State-of-health bucket; only 98–102 is kept |
| `cell_temperatures` | float | Bucketed cell temperature (°C) |
| `current_average` | float  | Bucketed pack current scaled to 277 Ah |
| `q_removed_ah`    | float  | Coulombs removed; converted to DoD = `q/277·100` |
| `cell_voltages_mean` | float | Cell voltage (mV) at this `(temp, current, DoD)` |

The repository ships an example at
`data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet`.

---

## Input data format

### CSV / Athena columns

| Column | Type | Description |
|---|---|---|
| `sequence_number` | int | Monotonically increasing row counter |
| `time_stamp` | datetime | Measurement timestamp |
| `temperature_battery` | float | Pack temperature (°C) |
| `current` | float | Pack current (A) — positive = charging |
| `charge_in` | float | Cumulative charge in (Coulombs) |
| `charge_out` | float | Cumulative charge out (Coulombs) |
| `soc` | float | BMS-reported SoC % (reference only) |
| `cell_voltage_1` … `cell_voltage_N` | float | Cell voltages (V) |

### MQTT payload

Messages arrive on topic `{gateway_id}/local/bem/response/{bem_id}/getLatestDatagram` and
carry a per-BEM datagram with `sequenceNumber`, `date`, `battery_SoC`, `bemCurrentInst` (mA),
`extTemp` (°C), and a `cells` array (each element with `cellVoltage` in mV and
`temperature` in °C).

The MQTT entry point uses a **separate parquet LUT** (temperature × current × DoD)
rather than the CSV OCV LUT used by the CSV/Athena CLIs — see `--lut-path` below.

---

## CLI usage

### CSV

```bash
uv run soc-estimate-csv \
  --csv         data/battery_measurements.csv \
  --lut-path    data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --capacity    277 \
  --num-cells   32 \
  --output-csv  results/soc_output.csv
```

### AWS Athena

Connection parameters are read from `.env` automatically; override with flags as needed:

```bash
uv run soc-estimate-athena \
  --serial      P34-000092 \
  --lut-path    data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --capacity    277 \
  --num-cells   32 \
  --output-csv  results/athena_soc.csv

# Override env vars inline:
uv run soc-estimate-athena --serial P34-000092 \
  --lut-path data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --capacity 277 --num-cells 32 \
  --database my_db --table telemetry --aws-profile staging \
  --s3-output s3://my-bucket/athena-results/
```

### MQTT (real-time)

**Best practice: run one process per gateway.** Each gateway gets its own broker
connection, reconnect/backoff loop, and process lifecycle, so a single broker hiccup or
a single gateway's misbehaviour can't take down the others. The CLI does accept
`--gateway` multiple times for convenience, but a supervisor (systemd, pm2, Docker,
etc.) restarting one `soc-estimate-mqtt` process per gateway is more robust at scale.

Most BEM-equipped gateways are also their own broker, so `--broker` and `--gateway`
are usually the same hostname:

```bash
uv run soc-estimate-mqtt \
  --gateway   gw-000105 \
  --num-bems  5 \
  --broker    gw-000105 \
  --port      1883 \
  --capacity  277 \
  --num-cells 32 \
  --lut-path  data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --log-dir   ./logs
```

With authentication:

```bash
uv run soc-estimate-mqtt --gateway gw-000105 --num-bems 5 \
  --broker mqtt.example.com --port 1883 \
  --username myuser --password mypass \
  --capacity 277 --num-cells 32 \
  --lut-path data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --log-dir ./logs
```

**Output:** one daily-rotating CSV per BEM at
`{log-dir}/{gateway}-{serial}-{bem_model}-{bem_id:02d}.log`. Columns include
`sequence_number`, `datestamp`, `soc` (BMS reported), `bemCurrentInst`, `extTemp`,
`cellVoltage_1..N`, and `soc_min_i`, `soc_max_i`, `soc_rolling_i` per cell.
`soc_rolling_i` fills in once a cell has ≥12 samples (≈12 ticks).

---

## State of Health (SoH)

Two extra CLIs estimate per-cell **State of Health** on top of the *same*
parquet-LUT SoC estimator used everywhere else (`bem_estimator.estimate_cell_soc`)
— the SoH layer just adds a charge-counter reference and a discharge gate. They
read from AWS Athena and use the same `.env` connection settings as
`soc-estimate-athena`.

**How SoH is derived (per cell):**

1. **Reference SoC from the charge counters** (`soc_nominal`) — the *q_top*
   algorithm latches the charge counter while a cell rests at the top of charge
   (≥3500 mV, idle, ≥10 ticks into the rest), forward-fills it, and converts the
   charge removed since that top into a SoC %.
2. **LUT SoC estimate** — every (cell voltage, bucketed temperature, bucketed
   current) sample is run through the shared estimator to get a `(soc_min,
   soc_max)` bracket; a short rolling mean of the midpoint gives
   `rolling_soc_mean`.
3. **SoH** — on discharge samples only (see `--current-threshold` below):

   ```
   soh = 100 − (soc_nominal − rolling_soc_mean)
   ```

   This is smoothed (long rolling mean) and forward-filled. Each cell's last
   forward-filled value is its reported SoH; the **pack SoH is the worst
   (lowest) cell**.

### The `--current-threshold` flag

SoH is only meaningful where the cell is under a known, steady load — the OCV→DoD
LUT curve is most reliable under solid discharge. `--current-threshold` is that
gate: a sample contributes to the SoH estimate only when

```
current_average ≤ current_threshold
```

`current_average` is the **charge-counter-derived current** (Δcharge per minute)
**scaled to the 277 Ah LUT reference** and floored to the nearest 5 A. It is
negative while discharging, so the threshold is a negative number. **The default
is `-100` A** (i.e. only count samples discharging at ≥100 A, reference-scaled).

| `--current-threshold` | Keeps samples that are… | Effect |
|---|---|---|
| `-150` | discharging ≥150 A | strictest — fewest, cleanest samples; may yield no SoH for lightly-used packs |
| `-100` *(default)* | discharging ≥100 A | balanced — solid discharge only |
| `-50`  | discharging ≥50 A  | more samples, noisier estimate |
| `0`    | any discharge      | most permissive; includes weak/near-idle discharge |

Because the value is scaled to 277 Ah, the same threshold means the same C-rate
across packs of different capacity — pass `--capacity` correctly and `-100`
behaves consistently whether the pack is 138 Ah or 277 Ah.

### Single battery

```bash
uv run soc-estimate-soh \
  --serial            BE20000157 \
  --capacity          277 \
  --num-cells         16 \
  --lut-path          data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --current-threshold -100 \
  --output-csv        results/soh_BE20000157.csv
```

Writes a per-cell-over-time CSV (`cell_number`, `time_stamp`, `soh_estimate_ffill`,
…) and logs the final SoH for each cell plus the pack worst-cell SoH. Tighten the
gate to `-150` for only the heaviest discharges, or loosen to `-50` to include
more (noisier) samples:

```bash
uv run soc-estimate-soh --serial BE20000157 --capacity 277 --num-cells 16 \
  --lut-path data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --current-threshold -50 \
  --start-date 2025-01-01 --end-date 2025-12-31
```

### Batch summary

Score many batteries from a spreadsheet (CSV or XLSX) with columns
`device_serial`, `capacity`, `cell_quantity`. Batteries sharing
`(capacity, cell_quantity)` are grouped and fetched `--chunk-size` serials at a
time, so only one chunk of raw data is in memory at once:

```bash
uv run soc-estimate-soh-batch \
  --input             batteries.csv \
  --lut-path          data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --num-cells         16 \
  --current-threshold -100 \
  --chunk-size        50 \
  --output-csv        results/soh_summary.csv
```

> `--num-cells` here is the cell count used when filtering Athena rows; per-row
> `cell_quantity` from the spreadsheet drives the grouping and the output width.

The summary CSV has one row per battery: `device_serial`, `capacity`,
`cell_quantity`, `soh_cell_1 … soh_cell_N`, `lowest_soh_per_pack` (the worst
cell), and `time_stamp_of_last_recording`.

---

## Python API

### CSV / Athena (parquet LUT, BEM-schema output)

```python
from soc_estimation import CSVDataSource, setup_logger
from soc_estimation.bem_estimator import (
    load_lut, make_lut_arrays, run_batch_estimation,
)

logger = setup_logger("my_app", "app.log")

lut = make_lut_arrays(load_lut(
    "data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
))

source = CSVDataSource("data/measurements.csv", num_cells=32, logger=logger)

written = run_batch_estimation(
    records=source.get_records(),
    num_cells=32,
    capacity_ah=277,
    lut=lut,
    output_csv="results/out.csv",
    logger=logger,
)
# Output is a single CSV — one row per tick — with the same columns as the
# MQTT BEM logs: soc_min_i, soc_max_i, soc_rolling_i for each cell.
```

### State of Health (SoH)

```python
import polars as pl
from soc_estimation import estimate_soh, summarise_soh
from soc_estimation.bem_estimator import load_lut, make_lut_arrays

lut = make_lut_arrays(load_lut(
    "data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet"
))

# df_raw: one row per tick with device_serial, sequence_number, time_stamp,
# current, charge_in, charge_out, temperature_battery, and cell_voltages as a
# per-row list of V (the Athena cell_data.cell_voltage array shape).
df_soh = estimate_soh(
    df_raw, lut, capacity_ah=277, num_cells=16, current_threshold=-100,
)
print(summarise_soh(df_soh, num_cells=16))   # final per-cell SoH + worst cell
```

`run_soh_batch(...)` wraps the chunked Athena fetch + summary used by
`soc-estimate-soh-batch`.

### Custom data source

```python
from typing import Iterator
from soc_estimation.data_sources import DataSource, BatteryDataRecord

class MySource(DataSource):
    def get_records(self) -> Iterator[BatteryDataRecord]:
        for row in my_database.fetch():
            yield BatteryDataRecord(
                sequence_number=row["seq"],
                time_stamp=row["ts"],
                temperature_battery=row["temp_c"],
                current=row["current_a"],
                cell_voltages={i: row[f"v{i}"] for i in range(1, 33)},
                charge_in=row["charge_in_c"],
                charge_out=row["charge_out_c"],
                soc_nominal=row["soc_pct"],
            )

# Then pass MySource().get_records() to run_batch_estimation(...)
```

---

## Output format

All three CLIs emit the same schema — **one row per tick** with per-cell SoC
columns laid out beside each other. (The MQTT logger writes one such file per
BEM into `--log-dir`; CSV/Athena write a single file to `--output-csv`.)

| Column                 | Type   | Description |
|------------------------|--------|-------------|
| `sequence_number`      | int    | Tick number from the source record |
| `datestamp`            | str    | Source timestamp |
| `soc`                  | float  | BMS-reported SoC % (reference only) |
| `bemCurrentInst`       | int    | Pack current in **mA** (signed; positive = charging) |
| `extTemp`              | float  | Pack/cell-bus temperature (°C) |
| `cellVoltage_1..N`     | int    | Per-cell voltage in **mV** |
| `soc_min_i`            | float  | Lower SoC bracket for cell *i* (this tick) |
| `soc_max_i`            | float  | Upper SoC bracket for cell *i* (this tick) |
| `soc_rolling_i`        | float  | 13-tick rolling mean of `(soc_min+soc_max)/2`; empty until ≥12 samples |

---

## Running tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=soc_estimation --cov-report=term-missing

# A single test file
uv run pytest tests/test_algorithm.py -v
```

The test suite uses synthetic fixture data in `tests/fixtures/` — no real battery data or AWS credentials are required.

---

## Development

```bash
# Lint
uv run ruff check soc_estimation/ tests/

# Format (auto-fix)
uv run ruff format soc_estimation/ tests/

# Format check only (as used in CI)
uv run ruff format --check soc_estimation/ tests/

# Update lockfile after editing dependencies in pyproject.toml
uv lock

# Re-sync dependencies (e.g. after a `git pull`)
uv sync --extra dev
```

---

## Project layout

```
soc_estimation/          # Installable package
├── bem_estimator.py     # Shared algorithm: load_lut, estimate_cell_soc,
│                        #   RollingSoc, run_batch_estimation (CSV/Athena)
├── mqtt_bem_logger.py   # MQTT entry point — subscribes per gateway, calls
│                        #   estimate_cell_soc on each datagram
├── soh_estimator.py     # Per-cell SoH (q_top reference + shared estimator):
│                        #   estimate_soh, summarise_soh, run_soh_batch
├── data_sources/
│   ├── base.py          # BatteryDataRecord + DataSource ABC
│   ├── csv_source.py    # Polars-backed CSV reader
│   └── athena_source.py # AWS Athena via awswrangler
├── core/ocv_algorithm.py # Legacy rest-window / OCV-CSV algorithm
├── engine.py            # Legacy SOCEstimationEngine (kept for library tests)
├── utils/               # config, lut_loader (CSV LUT), logging_utils
└── cli.py               # argparse entry points → bem_estimator / mqtt_bem_logger
                         #   / soh_estimator

archive/src/             # Legacy experimental scripts (not maintained)
tests/
├── fixtures/            # Synthetic CSV fixtures (committed)
├── test_algorithm.py    # Unit tests for legacy core algorithm
└── test_csv_integration.py  # End-to-end legacy engine tests
```
