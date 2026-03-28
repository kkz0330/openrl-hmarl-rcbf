# Grid Optimization Report (ma)

- Time: `2026-03-23 01:02:20 CST`
- Symbol: `KQ.m@INE.sc`
- Comparison table: `ma_KQ_m_INE_sc_grid_compare.csv`

## Best Parameters

- `fast_ma`: `10`
- `slow_ma`: `40`
- `breakout_window`: `20`
- `atr_window`: `14`
- `stop_atr`: `1.5`
- `rr`: `1.5`
- `risk_per_trade`: `0.01`

## Best Metrics

| Metric | Value |
|---|---:|
| Trades | 82 |
| Win Rate | 46.34% |
| Payoff Ratio | 1.537 |
| Profit Factor | 1.328 |
| Max Drawdown | -9.29% |
| Total Return | 11.14% |
| Annual Return (est.) | 276.45% |
| Sharpe (est.) | 4.773 |

## Top Parameter Sets

| rank | case_id | score | trades | win_rate | payoff | max_dd | ret | sharpe | params |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 3.552 | 82 | 46.34% | 1.537 | -9.29% | 11.14% | 4.773 | fast=10, slow=40, stop_atr=1.5, rr=1.5, risk=0.01 |
| 2 | 5 | 3.058 | 17 | 58.82% | 1.424 | -1.29% | 2.98% | 4.378 | fast=10, slow=40, stop_atr=2.0, rr=1.5, risk=0.005 |
| 3 | 4 | 2.945 | 67 | 41.79% | 1.860 | -7.63% | 10.10% | 3.834 | fast=10, slow=40, stop_atr=1.5, rr=2.0, risk=0.01 |
| 4 | 18 | 2.745 | 79 | 45.57% | 1.503 | -7.38% | 8.15% | 3.674 | fast=20, slow=40, stop_atr=1.5, rr=1.5, risk=0.01 |
| 5 | 6 | 2.607 | 47 | 48.94% | 1.434 | -3.76% | 6.37% | 3.475 | fast=10, slow=40, stop_atr=2.0, rr=1.5, risk=0.01 |
| 6 | 7 | 2.449 | 16 | 50.00% | 1.727 | -1.35% | 2.47% | 3.485 | fast=10, slow=40, stop_atr=2.0, rr=2.0, risk=0.005 |
| 7 | 8 | 2.056 | 40 | 42.50% | 1.772 | -3.73% | 4.96% | 2.734 | fast=10, slow=40, stop_atr=2.0, rr=2.0, risk=0.01 |
| 8 | 23 | 1.870 | 18 | 44.44% | 1.770 | -2.49% | 1.91% | 2.668 | fast=20, slow=40, stop_atr=2.0, rr=2.0, risk=0.005 |
| 9 | 20 | 1.827 | 67 | 38.81% | 1.897 | -8.44% | 6.08% | 2.424 | fast=20, slow=40, stop_atr=1.5, rr=2.0, risk=0.01 |
| 10 | 24 | 1.378 | 47 | 38.30% | 1.871 | -3.98% | 3.11% | 1.831 | fast=20, slow=40, stop_atr=2.0, rr=2.0, risk=0.01 |
| 11 | 28 | 1.283 | 57 | 40.35% | 1.707 | -5.99% | 3.70% | 1.670 | fast=20, slow=60, stop_atr=1.5, rr=2.0, risk=0.01 |
| 12 | 26 | 1.150 | 79 | 45.57% | 1.308 | -5.48% | 2.92% | 1.488 | fast=20, slow=60, stop_atr=1.5, rr=1.5, risk=0.01 |
| 13 | 3 | 0.841 | 43 | 41.86% | 1.531 | -2.66% | 1.05% | 1.084 | fast=10, slow=40, stop_atr=1.5, rr=2.0, risk=0.005 |
| 14 | 32 | 0.835 | 42 | 33.33% | 2.206 | -3.75% | 1.77% | 1.081 | fast=20, slow=60, stop_atr=2.0, rr=2.0, risk=0.01 |
| 15 | 22 | 0.730 | 51 | 43.14% | 1.420 | -4.69% | 1.47% | 0.921 | fast=20, slow=40, stop_atr=2.0, rr=1.5, risk=0.01 |
| 16 | 1 | 0.327 | 56 | 44.64% | 1.274 | -3.63% | 0.25% | 0.333 | fast=10, slow=40, stop_atr=1.5, rr=1.5, risk=0.005 |
| 17 | 12 | -0.124 | 65 | 35.38% | 1.789 | -6.10% | -0.79% | -0.184 | fast=10, slow=60, stop_atr=1.5, rr=2.0, risk=0.01 |
| 18 | 16 | -0.223 | 45 | 33.33% | 1.924 | -6.36% | -0.83% | -0.324 | fast=10, slow=60, stop_atr=2.0, rr=2.0, risk=0.01 |
| 19 | 10 | -0.403 | 83 | 40.96% | 1.394 | -7.65% | -1.45% | -0.556 | fast=10, slow=60, stop_atr=1.5, rr=1.5, risk=0.01 |
| 20 | 30 | -0.405 | 48 | 33.33% | 1.879 | -4.54% | -1.28% | -0.640 | fast=20, slow=60, stop_atr=2.0, rr=1.5, risk=0.01 |