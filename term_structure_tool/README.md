# 期限结构分析工具

该工具用于抓取某一类期货合约（同品种不同交割月）的收盘价，绘制期限结构曲线，并自动判断是升水还是贴水结构，同时给出研究型交易思路。

## 功能

- 抓取指定 `交易所 + 品种` 的未到期合约
- 读取各合约最新可得日线收盘价
- 绘制期限结构图（交割月-价格）
- 判断结构：升水 / 贴水 / 平坦混合
- 输出 CSV + PNG + Markdown 分析报告

## 运行

```bash
python term_structure_tool/term_structure_analyzer.py --exchange INE --product sc --user <账号> --password <密码>
```

可选参数：

- `--max-contracts`：最多分析多少个近月合约，默认 `12`
- `--output-dir`：输出目录，默认 `term_structure_tool/output`

也可使用环境变量：

```bash
set TQ_USER=<账号>
set TQ_PASSWORD=<密码>
python term_structure_tool/term_structure_analyzer.py --exchange INE --product sc
```

## 输出文件

默认输出到 `term_structure_tool/output`：

- `EXCHANGE_PRODUCT_term_structure.csv`
- `EXCHANGE_PRODUCT_term_structure.png`
- `EXCHANGE_PRODUCT_analysis.md`

## 结构判断规则

- 远近月价差（远月 - 近月）显著为正且曲线斜率为正：升水（Contango）
- 远近月价差显著为负且曲线斜率为负：贴水（Backwardation）
- 其他情况：平坦/混合结构

> 默认“显著”阈值是约 `1%`，可在代码中调整。

## 风险提示

策略建议仅用于研究，不构成投资建议。实盘需考虑：

- 流动性与冲击成本
- 保证金占用与强平风险
- 展期与换月规则
- 突发事件导致结构快速反转

