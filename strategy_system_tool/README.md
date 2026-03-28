# 策略回测系统（均线/突破 + 网格优化）

该目录提供完整的期货策略回测系统，包含：

- 开仓/平仓逻辑
- 资金管理（按风险比例定仓）
- 风控（止损、止盈、保证金占用、最大手数、手续费、滑点）
- 核心指标（胜率、赔率、最大回撤、收益率、夏普等）
- 参数网格优化（多参数对比表 + 最佳参数）

## 文件

- `backtest_system.py`：主程序
- `walk_forward_validation.py`：滚动样本外验证（每折先优化再测试）

## 依赖

```bash
pip install tqsdk pandas numpy matplotlib
```

## 单次回测

```bash
python strategy_system_tool/backtest_system.py \
  --strategy ma \
  --symbol KQ.m@INE.sc \
  --user <天勤账号> \
  --password <天勤密码>
```

## 网格优化

### 均线策略优化示例

```bash
python strategy_system_tool/backtest_system.py \
  --strategy ma \
  --symbol KQ.m@INE.sc \
  --bars 1200 \
  --optimize \
  --grid-fast-ma 10,20,30 \
  --grid-slow-ma 40,60,80 \
  --grid-stop-atr 1.5,2.0,2.5 \
  --grid-rr 1.5,2.0,2.5 \
  --grid-risk 0.005,0.01 \
  --user <天勤账号> \
  --password <天勤密码>
```

### 突破策略优化示例

```bash
python strategy_system_tool/backtest_system.py \
  --strategy breakout \
  --symbol KQ.m@INE.sc \
  --optimize \
  --grid-breakout-window 10,20,30 \
  --grid-stop-atr 1.5,2.0,2.5 \
  --grid-rr 1.5,2.0 \
  --grid-risk 0.005,0.01 \
  --user <天勤账号> \
  --password <天勤密码>
```

## 输出文件

默认目录：`strategy_system_tool/output`

单次回测：
- `*_trades.csv`
- `*_equity.csv`
- `*_chart.png`
- `*_report.md`

网格优化：
- `*_grid_compare.csv`（多参数对比表）
- `*_grid_report.md`（含Top组合与最佳参数）
- `*_best_trades.csv`
- `*_best_equity.csv`
- `*_best_chart.png`
- `*_best_report.md`

## 筛选规则

网格优化时会使用综合评分排序，并可过滤：

- `--min-trades-filter`：最小交易次数（默认 20）
- `--max-dd-filter`：最大回撤阈值（默认 0.3，即 -30%）

若过滤后为空，则回退使用全量最佳分数组合。

## 样本外 / 滚动验证（防过拟合）

该流程会对每一折执行：
1. 在训练窗做参数网格优化；
2. 将最佳参数应用到下一测试窗做样本外评估；
3. 汇总每折样本内/样本外指标，输出过拟合缺口。

示例：

```bash
python strategy_system_tool/walk_forward_validation.py \
  --strategy ma \
  --symbol KQ.m@INE.sc \
  --bars 1800 \
  --train-size 500 \
  --test-size 150 \
  --step-size 150 \
  --grid-fast-ma 10,20 \
  --grid-slow-ma 40,60 \
  --grid-stop-atr 1.5,2.0 \
  --grid-rr 1.5,2.0 \
  --grid-risk 0.005,0.01 \
  --user <天勤账号> \
  --password <天勤密码>
```

输出：
- `wf_*_folds.csv`：每折结果（含最佳参数、样本内/样本外收益和夏普）
- `wf_*_summary.md`：汇总结论（含收益缺口/夏普缺口）
- `wf_*_oos_equity.png`：样本外折间复利权益曲线

## 说明

- 本系统用于研究与教学，不构成投资建议。
- 建议先在较短样本和较小参数网格上验证，再扩大范围。
