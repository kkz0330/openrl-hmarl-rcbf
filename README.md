# INE Crude Dashboard (TqSdk)

This project reads INE crude futures (`sc`) market data from TqSdk and generates an HTML dashboard with:

- Current main-contract price
- Main-contract open price
- Contract OI change (lots) and OI change x multiplier (barrels)
- Term structure by delivery month
- Main-contract 1H candlestick chart
- Multi-model 24h forecasts (with validation metrics)

## Files

- `ine_crude_dashboard.py`: data fetch + forecasting + HTML rendering
- `ine_crude_dashboard.html`: generated dashboard output

## Install

```bash
pip install tqsdk pandas numpy plotly flask
```

## Run

```bash
python ine_crude_dashboard.py --user <TQ_USER> --password <TQ_PASSWORD>
```

Optional:

- `--output`: output HTML path (default: `ine_crude_dashboard.html`)

## Result Snapshot (Sample Run)

Run time: `2026-03-15 23:59:55 +08:00`  
Dashboard file: [`ine_crude_dashboard.html`](./ine_crude_dashboard.html)

### Market Snapshot

| Item | Value |
|---|---|
| Main Price | 786.80 |
| Main Open | 728.30 |
| Underlying of Main | INE.sc2604 |
| Strongest OI Increase | INE.sc2605, +5,232 lots, +5,232,000 barrels |
| Strongest OI Decrease | INE.sc2604, -6,638 lots, -6,638,000 barrels |
| Best Forecast Model | HoltLinear |

### Model Comparison (Validation + 24h Forecast)

| Model | MAE | RMSE | 24h Forecast | 95% Band |
|---|---:|---:|---:|---|
| LinearTrend | 37.293 | 40.230 | 779.99 | [747.21, 812.77] |
| EWMA | 73.349 | 76.396 | 747.99 | [706.12, 789.86] |
| AR1LogReturn | 38.298 | 41.355 | 830.40 | [797.69, 863.12] |
| HoltLinear | 26.222 | 32.512 | 904.34 | [847.15, 961.52] |

Note: This snapshot is only an example and will change with live market updates.

## Forecasting Methods

The dashboard runs 4 methods in parallel on the same close-price series:

1. `LinearTrend`: linear regression trend extrapolation
2. `EWMA`: exponential weighted moving-average level
3. `AR1LogReturn`: AR(1) on log returns, then rebuild price path
4. `HoltLinear`: Holt double exponential smoothing

Model comparison:

- Validation split: recent holdout slice from the latest 240 hourly bars
- Metrics: MAE and RMSE
- Best model: the one with minimum validation MAE

The page shows all model lines and a model metrics table.

## Notes

- Main chart uses `KQ.m@INE.sc` (main contract). Contract rollover can cause jumps.
- Forecasts are for analysis/demo only, not investment advice.
