# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Battery State of Charge (SOC) estimation package using Open-Circuit Voltage (OCV) lookup tables with rest-window gating. Supports per-cell tracking from three data sources: CSV files, AWS Athena, and real-time MQTT streams.

**Python 3.11+ required.** Package manager: `uv` (lock file present) or `pip`.

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black soc_estimation/

# Lint
ruff check soc_estimation/

# Type check
mypy soc_estimation/
```

## CLI Entry Points

```bash
soc-estimate-csv --csv data/measurements.csv --lut_csv data/ocv_lut.csv --capacity 277 --num_cells 32 --output_csv results/out.csv
soc-estimate-athena --serial P34-000092 --lut_csv data/ocv_lut.csv --capacity 277 --num_cells 32 --database battery_data --table measurements --aws_profile infra_dev --output_csv results/out.csv
soc-estimate-mqtt --broker mqtt.example.com --port 1883 --gateway gw-000027 --bem_ids 1 2 3 4 --lut_csv data/ocv_lut.csv --capacity 277 --num_cells 32
```

## Architecture

### Two Codebases in This Repo

- **`soc_estimation/`** — The installable package (new, modular). This is the primary codebase to work with.
- **`src/`** — 35+ legacy experimental/analysis scripts. These are standalone files that may share logic but are not part of the package. Use for reference only.

### Package Structure (`soc_estimation/`)

```
soc_estimation/
├── core/ocv_algorithm.py     # Core per-cell estimation algorithm
├── data_sources/
│   ├── base.py               # Abstract DataSource + BatteryDataRecord
│   ├── csv_source.py         # Polars-based CSV reader
│   ├── athena_source.py      # AWS Athena via awswrangler
│   └── mqtt_source.py        # Async MQTT via aiomqtt
├── utils/
│   ├── lut_loader.py         # Loads OCV lookup table CSV
│   └── logging_utils.py      # Rotating file + stderr logger
├── engine.py                 # Orchestration: reads records, runs algorithm, writes output
└── cli.py                    # argparse CLIs wrapping engine.py
```

### Data Flow

```
DataSource.get_records() → BatteryDataRecord
    → SOCEstimationEngine.process_all_records()
        → RestWindowState (pack-level current monitor)
        → process_cell_tick() per cell
            → OCV LUT (linear interpolation)
        → CellTickResult
    → output CSV
```

### Key Data Structures

**Input** — `BatteryDataRecord`:
- `sequence_number`, `time_stamp`, `temperature_battery`
- `current` (A, positive = charging)
- `cell_voltages: dict[int, float]` — cell index → voltage in V
- `charge_in`, `charge_out` (Coulombs, cumulative)
- `soc_nominal` (BMS reference %)

**Output** — one row per cell per tick:
- `sequence_number`, `cell_idx`, `time_stamp`, `soc_nominal`
- `mode`: `LUT` | `CC` | `UNINIT` | `SKIP`
- `soc_min`, `soc_max`, `soc_mid`, `soc_hybrid`

**OCV LUT CSV**: two columns — `soc` (0–100%) and `voltage` (mV).

**Input CSV required columns**: `sequence_number`, `time_stamp`, `temperature_battery`, `current`, `charge_in`, `charge_out`, `soc`, `cell_voltage_1`…`cell_voltage_N`.

### Algorithm (`core/ocv_algorithm.py`)

`process_cell_tick()` runs these steps per cell per tick:
1. Validate voltage → `SKIP` if invalid
2. Check `RestWindowState`: pack current within ±`rest_current_a` for `rest_min_ticks` consecutive ticks → `CC` (coulomb counting) if not resting
3. (Optional) Voltage delta gate — skip LUT reset if voltage hasn't moved enough
4. OCV linear interpolation: binary search + interpolate on sorted LUT
5. (Optional) Voltage tolerance gate — skip if voltage is far from nearest LUT point
6. LUT reset → update `CellState`; falls back to `UNINIT` until first LUT reset

**Key algorithm parameters** (all passed to `SOCEstimationEngine`):

| Parameter | Default | Notes |
|---|---|---|
| `rest_current_a` | 1.0 | ±A threshold to qualify rest window |
| `rest_min_ticks` | 5 | Consecutive ticks @ ~60s each |
| `voltage_tolerance_mv` | 0.0 | 0 = disabled |
| `voltage_delta_min_mv` | 0.0 | 0 = disabled |
| `capacity_ah` | required | Pack capacity in Ah |
| `num_cells` | required | Cells per pack |

### Adding a Data Source

Subclass `DataSource` from `soc_estimation.data_sources.base` and implement `get_records() -> Iterator[BatteryDataRecord]`. For async (MQTT-style), also implement `get_records_async()`. Pass the instance directly to `SOCEstimationEngine`.

## Dependencies

Core: `polars`, `pandas`, `numpy`, `boto3`, `awswrangler` (Athena), `aiomqtt` (MQTT async).  
Dev: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`.
