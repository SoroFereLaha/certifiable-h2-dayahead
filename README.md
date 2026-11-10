# Certifiable hydrogen commitments under hourly matching

Code, input data and results for the paper "Forecasting Certifiable Hydrogen for Day-Ahead Delivery under
Hourly Matching Rules" (F. Soro, N. Rouky, M. Fri, O. Benmoussa, AICSET'26, Springer).

A hydrogen producer buys the output of a wind farm and runs an electrolyser. At 18:15 on the day before
delivery, it must commit to the amount of electricity, and hence hydrogen, that will count as renewable under
the EU hourly-matching rule (RFNBO). The paper forecasts the 5th percentile of this certifiable energy from
public Danish (DK1) forecasts and compares simple baselines, a recently calibrated reference and learned
quantile models.

## Methods and stages

| Name | Commitment |
|---|---|
| B0 | TSO day-ahead estimate, no safety margin |
| B1 | TSO estimate minus one conformal margin |
| B1′ | TSO estimate minus a margin per quintile of the estimate |
| C1 | Gradient-boosting quantile model (stage 1) |
| C2 | C1 with weather predictors |
| R | B1′ calibrated on the last 12 months only (reference of stage 2) |
| C3 | Model chosen in stage 2: recency-weighted linear quantile regression |

Stage 1 compares C1 with the baselines on the test period (hypotheses H1–H5). Stage 2 chooses C3 on
development data, then tests it once against R on 92 new days, July–September 2026 (H6, H7). The scripts
named `posthoc_*` run the analyses added after that test. H4 replays the commitments on a local blockchain.

## Layout

| Path | Content |
|---|---|
| `src/core.py` | Shared definitions: data loading, certifiable energy $Q_D$, baselines, model C1, conformal margin, pinball loss, block bootstrap |
| `src/stage1.py` | Stage 1: H1–H3, sensitivity runs, C2 (H5) and the optional registry replay (H4) |
| `src/stage2_common.py` | Stage 2: candidate grid, development folds F1–F3, shared helpers |
| `src/stage2_selection.py` | Stage 2: choice of C3 on the development folds |
| `src/stage2_final_test.py` | Stage 2: one run of C3 and the baselines on the new days |
| `src/posthoc_qrf.py`, `src/posthoc_copula.py` | Quantile regression forest and Gaussian-copula scenarios |
| `src/posthoc_backtest.py` | C3 and R refitted as of 2023 and as of 2024 |
| `src/posthoc_penalties.py` | Bonferroni and Holm corrections; other penalty ratios |
| `src/posthoc_forecast_bias.py` | Yearly bias of the TSO day-ahead forecast |
| `src/registry.py`, `src/CommitmentRegistry.sol` | H4: commitments and settlements replayed on a local EVM (Anvil) |
| `src/weather.py` | Weather predictors for C2 |
| `src/fetch_energinet.py`, `src/snapshot_dayahead.sh` | Download of the Energinet series; snapshots of the day-ahead forecast |
| `src/checks/` | Data checks: forecast revisions, missing hours, prices |
| `src/test_core.py` | Unit tests |
| `data/raw/`, `data/weather/` | Input series, each with `manifest.json` (request URL, row count and SHA-256 of each file) |
| `data/forecast_snapshots/` | Day-ahead forecast snapshot read by `src/checks/check_forecast_snapshots.py` |
| `results/` | Result files used in the paper, one per script |
| `results/reproducibility/` | A second run of stage 1 and a numpy-only recomputation of the stage-2 metrics |

## Environment

Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1 (`pip install -r requirements.txt`).
The registry replay also needs web3 7.16, solc 0.8.35 and Anvil 1.7.

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

Scripts that take `--output` stop if the file already exists. Bootstrap intervals use 7-day blocks, 5000
resamples and the fixed seed in `core.py`, so a new run gives the same numbers.

## Data sources

- Energinet, Energi Data Service: `Forecasts_Hour`, `ProductionConsumptionSettlement`, `Elspotprices`,
  `DayAheadPrices` (public API, no account). `data/raw/manifest.json` lists the request URLs.
- Open-Meteo Previous Model Runs API, ECMWF IFS 0.25° (CC BY 4.0). `data/weather/manifest.json` lists the
  request URLs.

The files are copies of these public series as downloaded for the study; the providers' terms of use apply.
`src/fetch_energinet.py` downloads the Energinet series again.

## Licence

Code: MIT (see `LICENSE`). Data files remain under the terms of their providers (see above).
