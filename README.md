# soc-estimation

Battery State-of-Charge (SoC) estimation using an OCV (Open-Circuit Voltage) lookup table with intelligent rest-window gating. Supports per-cell tracking across three data sources: CSV files, AWS Athena, and real-time MQTT streams.

[![CI](https://github.com/YOUR_ORG/state-of-charge-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/state-of-charge-estimation/actions/workflows/ci.yml)

---

## How it works

The estimator processes battery telemetry one tick at a time (typically 60 s cadence) and produces a SoC estimate for every cell in the pack.

**Per-cell decision logic (6 steps):**

1. **Validate** — skip missing or zero cell voltages (`SKIP` mode).
2. **Rest window gate** — the pack current must stay within `±rest_current_a` for `rest_min_ticks` consecutive ticks before an OCV reset is allowed. If not at rest, fall back to Coulomb Counting (`CC`) or return uninitialised (`UNINIT`).
3. *(Optional)* **Voltage-delta gate** — require `|Δv| ≥ voltage_delta_min_mv` to prevent resets on a flat curve.
4. **OCV interpolation** — look up SoC via binary-search + linear interpolation on the LUT.
5. *(Optional)* **Voltage-tolerance gate** — reject if the measured voltage is more than `voltage_tolerance_mv` from the nearest LUT point.
6. **LUT reset** — accept the new SoC and reset the rest counter so a fresh rest window is required before the next reset.

---

## Prerequisites

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) **or** `pip`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_ORG/state-of-charge-estimation.git
cd state-of-charge-estimation

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with plain pip
pip install -e ".[dev]"
```

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

The LUT is a CSV with two columns:

| Column    | Type    | Description             |
|-----------|---------|-------------------------|
| `soc`     | float   | State of charge (0–100 %) |
| `voltage` | float   | Cell OCV in **mV**      |

Example (`data/ocv_lut.csv`):

```csv
soc,voltage
0,3000
10,3300
...
100,4100
```

Rows can be in any order; the loader sorts them by voltage automatically.

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

Messages arrive on topic `{gateway_id}/local/bem/response/{bem_id:02d}/getLatestDatagram`.  
Current is read from `bemCurrentInst` (mA, converted to A automatically).

---

## CLI usage

### CSV

```bash
soc-estimate-csv \
  --csv          data/battery_measurements.csv \
  --lut_csv      data/ocv_lut.csv \
  --capacity     277 \
  --num_cells    32 \
  --output_csv   results/soc_output.csv
```

Optional tuning flags (same for all three CLIs):

```
--rest_current_a     1.0   # ±A threshold for rest window (default 1.0)
--rest_min_ticks     5     # ticks required for rest (default 5 = 5 min)
--voltage_tolerance_mv 0   # reject if >N mV from nearest LUT point (0=off)
--voltage_delta_min_mv 0   # require voltage to change by ≥N mV (0=off)
--log_file soc.log
```

### AWS Athena

Connection parameters are read from `.env` automatically; override with flags as needed:

```bash
soc-estimate-athena \
  --serial    P34-000092 \
  --lut_csv   data/ocv_lut.csv \
  --capacity  277 \
  --num_cells 32 \
  --output_csv results/athena_soc.csv

# Override env vars inline:
soc-estimate-athena --serial P34-000092 --lut_csv data/ocv_lut.csv \
  --capacity 277 --num_cells 32 \
  --database my_db --table telemetry --aws_profile staging \
  --s3_output s3://my-bucket/athena-results/
```

### MQTT (real-time)

```bash
soc-estimate-mqtt \
  --broker    mqtt.example.com \
  --port      1883 \
  --gateway   gw-000027 \
  --gateway   gw-000105 \
  --bem_ids   1 2 3 4 \
  --lut_csv   data/ocv_lut.csv \
  --capacity  277 \
  --num_cells 32

# With TLS and authentication:
soc-estimate-mqtt --broker mqtt.example.com --port 8883 \
  --use_tls --username myuser --password mypass \
  --gateway gw-000027 --bem_ids 1 2 3 4 \
  --lut_csv data/ocv_lut.csv --capacity 277 --num_cells 32
```

---

## Python API

### CSV

```python
from soc_estimation import CSVDataSource, SOCEstimationEngine, setup_logger

logger = setup_logger("my_app", "app.log")

source = CSVDataSource("data/measurements.csv", num_cells=32, logger=logger)
engine = SOCEstimationEngine(
    data_source=source,
    num_cells=32,
    capacity_ah=277,
    ocv_lut_path="data/ocv_lut.csv",
    rest_current_a=1.0,
    rest_min_ticks=5,
    logger=logger,
)
results = engine.process_all_records(output_csv="results/out.csv")
# results → list of dicts, one per (cell, tick)
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
                # delta_charge is computed automatically if you leave it None
                # and want the engine to handle it, or supply it directly:
                delta_charge=row.get("delta_charge_c"),
            )
```

---

## Output format

One row per cell per tick:

| Column | Type | Description |
|---|---|---|
| `sequence_number` | int | Tick number |
| `cell_idx` | int | Cell index (1-based) |
| `mode` | str | `LUT` / `CC` / `UNINIT` / `SKIP` |
| `soc_min` | float | Lower bracket SoC (LUT mode only) |
| `soc_max` | float | Upper bracket SoC (LUT mode only) |
| `soc_mid` | float | Mid bracket SoC (LUT mode only) |
| `soc_hybrid` | float | Final SoC estimate (null if UNINIT) |
| `time_stamp` | str | Measurement timestamp |
| `soc_nominal` | float | BMS reference SoC |

---

## Running tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=soc_estimation --cov-report=term-missing

# A single test file
pytest tests/test_algorithm.py -v
```

The test suite uses synthetic fixture data in `tests/fixtures/` — no real battery data or AWS credentials are required.

---

## Development

```bash
# Lint
ruff check soc_estimation/ tests/

# Format (auto-fix)
ruff format soc_estimation/ tests/

# Format check only (as used in CI)
ruff format --check soc_estimation/ tests/
```

---

## Project layout

```
soc_estimation/          # Installable package
├── core/
│   └── ocv_algorithm.py # Algorithm: voltage_to_soc, process_cell_tick,
│                        #   CellState, RestWindowState
├── data_sources/
│   ├── base.py          # BatteryDataRecord + DataSource ABC
│   ├── csv_source.py    # Polars-backed CSV reader
│   ├── athena_source.py # AWS Athena via awswrangler
│   └── mqtt_source.py   # Async real-time MQTT
├── utils/
│   ├── config.py        # Env var / .env loading (AthenaConfig)
│   ├── lut_loader.py    # load_ocv_lut()
│   └── logging_utils.py # Rotating file + stderr logger
├── engine.py            # SOCEstimationEngine — orchestrates everything
└── cli.py               # argparse entry points

archive/src/             # Legacy experimental scripts (not maintained)
tests/
├── fixtures/            # Synthetic CSV fixtures (committed)
├── test_algorithm.py    # Unit tests for core algorithm
└── test_csv_integration.py  # End-to-end CSV → engine → results
```
