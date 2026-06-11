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
├── data_sources/
│   ├── base.py          # BatteryDataRecord + DataSource ABC
│   ├── csv_source.py    # Polars-backed CSV reader
│   └── athena_source.py # AWS Athena via awswrangler
├── core/ocv_algorithm.py # Legacy rest-window / OCV-CSV algorithm
├── engine.py            # Legacy SOCEstimationEngine (kept for library tests)
├── utils/               # config, lut_loader (CSV LUT), logging_utils
└── cli.py               # argparse entry points → bem_estimator / mqtt_bem_logger

archive/src/             # Legacy experimental scripts (not maintained)
tests/
├── fixtures/            # Synthetic CSV fixtures (committed)
├── test_algorithm.py    # Unit tests for legacy core algorithm
└── test_csv_integration.py  # End-to-end legacy engine tests
```
