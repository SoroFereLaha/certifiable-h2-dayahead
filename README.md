# Certifiable hydrogen commitments under hourly matching

Code, input data and results for the paper "Forecasting Certifiable Hydrogen for Day-Ahead Delivery under
Hourly Matching Rules" (F. Soro, N. Rouky, M. Fri, O. Benmoussa, AICSET'26, Springer).

The study forecasts, one day ahead, how much hydrogen a wind-powered electrolyser can certify under the EU
hourly-matching rule, and compares simple baselines, a recently calibrated reference and learned quantile models
on Danish (DK1) data.

## Layout

| Path | Content |
|---|---|
| `src/core.py` | Shared definitions: data loading, certifiable energy $Q_D$, baselines B0/B1/B1′, model C1, conformal shift, pinball loss, block bootstrap |
| `src/stage1.py` | Stage 1: C1 against the baselines on the test period (H1–H3, sensitivities, C2/H5, optional registry replay H4) |
| `src/stage2_common.py` | Stage 2: candidate grid, development folds F1–F3, shared helpers |
| `src/stage2_selection.py` | Stage 2: choice of C3 on the development folds |
| `src/stage2_final_test.py` | Stage 2: one pass of C3 and the baselines on the new days (Jul–Sep 2026) |
| `src/posthoc_qrf.py`, `src/posthoc_copula.py` | Quantile regression forest and Gaussian-copula scenarios, added after the final test |
| `src/posthoc_backtest.py` | C3 and R refitted as of 2023 and 2024 |
| `src/posthoc_penalties.py` | Bonferroni/Holm corrections and other penalty ratios |
| `src/posthoc_forecast_bias.py` | Yearly bias of the TSO day-ahead forecast |
| `src/registry.py`, `src/CommitmentRegistry.sol` | H4: commitments and settlements replayed on a local EVM (Anvil) |
| `src/weather.py` | Weather predictors for C2 |
| `src/fetch_energinet.py`, `src/snapshot_dayahead.sh` | Download of the Energinet series; snapshots of the day-ahead forecast |
| `src/checks/` | Data checks: forecast revisions, missing hours, prices |
| `src/test_core.py` | Unit tests |
| `data/raw/`, `data/weather/` | Input series, each with `manifest.json` (request URL, row count, SHA-256 of each file) |
| `data/forecast_snapshots/` | Day-ahead forecast snapshot used by `checks/check_forecast_snapshots.py` |
| `results/` | All result files used in the paper (one per script) |
| `results/reproducibility/` | A full rerun of stage 1 and an independent numpy-only recomputation of stage 2 |

## Environment

Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1 (`pip install -r requirements.txt`).
The registry replay additionally needs web3 7.16, solc 0.8.35 and Anvil 1.7.

## Reproduction

Run from `src/`:

```bash
python3 -m unittest test_core                       # unit tests (EVM test: RUN_EVM_TESTS=1, needs Anvil)
python3 stage1.py --output ../results/new_stage1.json [--registry]
python3 stage2_selection.py --output ../results/new_selection.json
python3 stage2_final_test.py --selection ../results/stage2_selection.json --output ../results/new_final_test.json
python3 posthoc_qrf.py --output ../results/new_qrf.json
python3 posthoc_copula.py --output ../results/new_copula.json
python3 posthoc_backtest.py --output ../results/new_backtest.json
python3 posthoc_penalties.py                        # writes results/posthoc_penalties.json
python3 posthoc_forecast_bias.py                    # writes results/posthoc_forecast_bias.json
python3 checks/check_missing_hours.py               # each check writes results/data_checks/<name>.json
```

Scripts with `--output` refuse to overwrite an existing file. Bootstrap intervals use 7-day blocks, 5000
resamples and the fixed seed in `core.py`.

## Data sources

- Energinet, Energi Data Service: `Forecasts_Hour`, `ProductionConsumptionSettlement`, `Elspotprices`,
  `DayAheadPrices` (public API, no account). Request URLs are listed in `data/raw/manifest.json`.
- Open-Meteo Previous Model Runs API, ECMWF IFS 0.25° (CC BY 4.0). Request URLs are in
  `data/weather/manifest.json`.

The files are copies of these public series as downloaded for the study; please refer to the providers'
terms of use. `src/fetch_energinet.py` re-downloads the Energinet series.

## Licence

Code: MIT (see `LICENSE`). Data files remain under the terms of their providers (see above).
