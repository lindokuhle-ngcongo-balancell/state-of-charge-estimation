# SoH `--current-threshold` — reference & sensitivity

This note documents how the `--current-threshold` gate behaves in the SoH
estimators (`soc-estimate-soh`, `soc-estimate-soh-batch`) and why **−100 A is
the recommended default**. It's backed by a real sweep on a single battery.

## What the threshold does

SoH is derived per cell as

```
soh = 100 − (soc_nominal − rolling_soc_mean)
```

…but only on **discharge samples that clear the gate**:

```
current_average ≤ current_threshold
```

`current_average` is the charge-counter-derived current (Δcharge per minute),
**scaled to the 277 Ah LUT reference** and floored to the nearest 5 A. It is
negative while discharging, so the threshold is a negative number. Because it's
reference-scaled, the *same* threshold means the *same* C-rate across packs of
different capacity (pass `--capacity` correctly).

The gate exists because the OCV→DoD LUT curve is most reliable under a known,
steady load. Too loose and you fold in light/near-idle discharge (noisy); too
strict and you starve the estimate of samples.

## Sweep — BE19090008 (277 Ah, 16 cells)

Full history: **3,863,057 raw rows** from Athena, scored through the shared
`bem_estimator` SoC estimator. Same raw data, three gates.

### Sample count

| Threshold | Qualifying rows | vs default |
|---|---|---|
| `-50`  | 3,471,896 | ~12.6× more |
| `-100` *(default)* | 276,232 | — |
| `-150` | 68,410 | ~4× fewer |

### Per-cell SoH (%)

| Cell | −50 | −100 | −150 | | Cell | −50 | −100 | −150 |
|---|---|---|---|---|---|---|---|---|
| 1 | 58.1 | 56.5 | 69.5 | | 9 | 62.0 | 59.2 | 73.4 |
| 2 | 55.2 | 53.9 | 68.1 | | 10 | 60.0 | 57.9 | 72.6 |
| 3 | 59.0 | 57.7 | 72.9 | | 11 | 50.8 | 50.6 | 67.1 |
| 4 | 58.3 | 58.6 | 74.0 | | 12 | 56.7 | 57.9 | 76.5 |
| 5 | 61.4 | 57.2 | 71.2 | | 13 | 59.5 | 59.9 | 77.3 |
| **6** | **21.5** | **28.2** | **33.2** | | 14 | 56.6 | 54.7 | 70.4 |
| 7 | 59.5 | 57.6 | 71.6 | | **15** | **25.3** | **33.0** | **46.7** |
| 8 | 64.9 | 63.5 | 77.0 | | 16 | 52.5 | 51.9 | 65.6 |

**Pack (worst-cell) SoH:** −50 → 21.5% · −100 → 28.2% · −150 → 33.2%

## Findings

1. **The threshold shifts the absolute level, not just the sample count.**
   −50 and −100 are close (within ~1–3 pts for healthy cells), but **−150 reads
   ~12–15 pts higher across the board.** Under the heaviest discharge the cell
   voltage sits lower on the OCV→DoD curve, so the LUT estimate and the IR term
   land differently — the SoH baseline is *not* threshold-invariant.
   → **Don't compare a `-150` number against a `-100` number.** Pick one gate and
   keep it fixed for any cross-battery or over-time comparison.

2. **The cell *ranking* is rock-solid.** Cells 6 and 15 are the two worst at
   every threshold; the healthy cluster (~55–75%) stays intact. Whatever gate you
   choose, the tool flags the same degraded cells.

3. **−100 A is the sensible default.** 276k samples is plenty for a stable
   estimate. `-50` (3.5M rows, mostly light/noisy discharge) drags the worst
   cells lower without adding confidence; `-150` is starved (68k rows) and biased
   high.

## Recommendation

- **Use `-100` (the default)** unless you have a specific reason not to.
- Loosen toward `-50` only for lightly-cycled packs that never see heavy
  discharge (accept the added noise).
- Tighten toward `-150` only when you have abundant heavy-discharge data and want
  the cleanest possible samples — and remember the level shifts up.
- Keep the threshold **constant** across any comparison.

## Reproduce

```bash
# Single battery, default gate
uv run soc-estimate-soh \
  --serial BE19090008 --capacity 277 --num-cells 16 \
  --lut-path data/lut/voltage_vs_soh_vs_temperature_vs_q_removed_vs_current_average_277ah.parquet \
  --current-threshold -100 \
  --output-csv results/soh_BE19090008_thr-100.csv
```

Raw outputs for this sweep (gitignored — runtime data):

| File | Rows |
|---|---|
| `results/soh_BE19090008_thr-50.csv`  | 3,471,896 |
| `results/soh_BE19090008_thr-100.csv` | 276,232 |
| `results/soh_BE19090008_thr-150.csv` | 68,410 |
| `results/soh_BE19090008_threshold_sweep.csv` | 48 (tidy summary: threshold × cell) |
