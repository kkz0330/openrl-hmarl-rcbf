import argparse
import ast
from pathlib import Path

import pandas as pd

from term_structure_tool.md_to_pdf import convert_md_to_pdf


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}"


def _fmt_num(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def _table_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    x = df.head(max_rows).copy()
    headers = list(x.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in x.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成策略说明文档 PDF（含回测、网格优化、样本外验证结果）")
    parser.add_argument("--output-dir", default="strategy_system_tool/output", help="结果目录")
    parser.add_argument("--stem", default="ma_KQ_m_INE_sc", help="结果文件前缀")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    stem = args.stem

    grid_csv = output_dir / f"{stem}_grid_compare.csv"
    best_chart = output_dir / f"{stem}_best_chart.png"
    wf_folds_csv = output_dir / f"wf_{stem}_folds.csv"
    wf_equity_png = output_dir / f"wf_{stem}_oos_equity.png"

    if not grid_csv.exists():
        raise SystemExit(f"未找到网格结果：{grid_csv}")
    if not wf_folds_csv.exists():
        raise SystemExit(f"未找到滚动验证结果：{wf_folds_csv}")

    grid_df = pd.read_csv(grid_csv, encoding="utf-8-sig")
    wf_df = pd.read_csv(wf_folds_csv, encoding="utf-8-sig")
    if grid_df.empty:
        raise SystemExit("网格结果为空")
    if wf_df.empty:
        raise SystemExit("滚动验证结果为空")

    best = grid_df.sort_values("score", ascending=False).iloc[0]
    try:
        best_params_text = best.get("best_params", "")
        if best_params_text:
            best_params = ast.literal_eval(best_params_text)
        else:
            best_params = {
                "fast_ma": int(best["fast_ma"]) if "fast_ma" in best else None,
                "slow_ma": int(best["slow_ma"]) if "slow_ma" in best else None,
                "breakout_window": int(best["breakout_window"]) if "breakout_window" in best else None,
                "stop_atr": float(best["stop_atr"]) if "stop_atr" in best else None,
                "rr": float(best["rr"]) if "rr" in best else None,
                "risk_per_trade": float(best["risk_per_trade"]) if "risk_per_trade" in best else None,
            }
    except Exception:
        best_params = {}

    wf_summary = {
        "folds": int(len(wf_df)),
        "oos_positive_ratio": float((wf_df["oos_total_return"] > 0).mean()),
        "oos_mean_return": float(wf_df["oos_total_return"].mean()),
        "oos_median_return": float(wf_df["oos_total_return"].median()),
        "is_mean_return": float(wf_df["is_total_return"].mean()),
        "is_mean_sharpe": float(wf_df["is_sharpe"].mean()),
        "oos_mean_sharpe": float(wf_df["oos_sharpe"].mean()),
        "return_gap": float(wf_df["is_total_return"].mean() - wf_df["oos_total_return"].mean()),
        "sharpe_gap": float(wf_df["is_sharpe"].mean() - wf_df["oos_sharpe"].mean()),
    }

    overfit_note = "过拟合风险可控"
    if wf_summary["return_gap"] > 0.01 or wf_summary["sharpe_gap"] > 1.0:
        overfit_note = "存在过拟合风险（样本内显著优于样本外）"
    if wf_summary["oos_positive_ratio"] < 0.5:
        overfit_note += "，且样本外正收益折占比偏低"

    now = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    md_path = output_dir / f"{stem}_strategy_doc.md"
    pdf_path = output_dir / f"{stem}_strategy_doc.pdf"

    best_params_lines = []
    if best_params:
        for k, v in best_params.items():
            if v is None:
                continue
            best_params_lines.append(f"- `{k}`: `{v}`")
    else:
        best_params_lines.append("- 见网格对比表第一行（最佳 score）")

    top_cols = [
        c
        for c in ["case_id", "score", "trades", "win_rate", "payoff_ratio", "max_drawdown", "total_return", "sharpe", "fast_ma", "slow_ma", "stop_atr", "rr", "risk_per_trade"]
        if c in grid_df.columns
    ]
    top_df = grid_df[top_cols].sort_values("score", ascending=False).head(10).copy()
    for col in ["win_rate", "max_drawdown", "total_return"]:
        if col in top_df.columns:
            top_df[col] = top_df[col].map(lambda x: f"{x:.2%}")
    for col in ["score", "payoff_ratio", "sharpe", "stop_atr", "rr", "risk_per_trade"]:
        if col in top_df.columns:
            top_df[col] = top_df[col].map(lambda x: f"{x:.3f}")

    wf_preview = wf_df[
        [
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "is_total_return",
            "oos_total_return",
            "is_sharpe",
            "oos_sharpe",
        ]
    ].copy()
    for col in ["is_total_return", "oos_total_return"]:
        wf_preview[col] = wf_preview[col].map(lambda x: f"{x:.2%}")
    for col in ["is_sharpe", "oos_sharpe"]:
        wf_preview[col] = wf_preview[col].map(lambda x: f"{x:.3f}")

    lines = [
        "# 量化交易策略说明文档",
        "",
        f"- 生成时间：`{now}`",
        f"- 策略前缀：`{stem}`",
        "",
        "## 1. 策略框架",
        "",
        "- 策略类型：均线 / 突破（本次结果为均线策略样例）。",
        "- 开仓：",
        "- 均线策略：快线 > 慢线做多，快线 < 慢线做空。",
        "- 突破策略：突破前高做多，跌破前低做空。",
        "- 平仓：反向信号、止损、止盈、或样本结束强平。",
        "- 资金管理：每笔风险 = 权益 * 风险比例，结合 ATR 与合约乘数定仓。",
        "- 风控约束：止损 ATR 倍数、收益风险比止盈、最大保证金占用、最大手数、手续费、滑点。",
        "",
        "## 2. 核心量化指标定义",
        "",
        "- 胜率 = 盈利笔数 / 总笔数",
        "- 赔率 = 平均盈利 / 平均亏损（绝对值）",
        "- 最大回撤 = 权益曲线峰谷最大跌幅",
        "- 收益率 = 期末权益 / 初始资金 - 1",
        "- 夏普比 = 年化后单位波动收益",
        "",
        "## 3. 参数网格优化结果",
        "",
        "### 3.1 最佳参数（按综合评分）",
        "",
        *best_params_lines,
        "",
        "### 3.2 最佳参数对应指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 交易次数 | {int(best['trades'])} |",
        f"| 胜率 | {_fmt_pct(float(best['win_rate']))} |",
        f"| 赔率 | {_fmt_num(float(best['payoff_ratio']))} |",
        f"| 最大回撤 | {_fmt_pct(float(best['max_drawdown']))} |",
        f"| 收益率 | {_fmt_pct(float(best['total_return']))} |",
        f"| 夏普比 | {_fmt_num(float(best['sharpe']))} |",
        "",
        "### 3.3 网格Top10参数对比",
        "",
        _table_markdown(top_df, max_rows=10),
        "",
    ]

    if best_chart.exists():
        lines.extend(
            [
                "### 3.4 最佳参数权益表现",
                "",
                f"![best_chart]({best_chart.name})",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 样本外/滚动验证结果（Walk-Forward）",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 折数 | {wf_summary['folds']} |",
            f"| 样本外正收益折占比 | {_fmt_pct(wf_summary['oos_positive_ratio'])} |",
            f"| 样本外平均收益率（每折） | {_fmt_pct(wf_summary['oos_mean_return'])} |",
            f"| 样本外中位收益率（每折） | {_fmt_pct(wf_summary['oos_median_return'])} |",
            f"| 样本内平均收益率（每折） | {_fmt_pct(wf_summary['is_mean_return'])} |",
            f"| 样本内平均夏普 | {_fmt_num(wf_summary['is_mean_sharpe'])} |",
            f"| 样本外平均夏普 | {_fmt_num(wf_summary['oos_mean_sharpe'])} |",
            f"| 收益缺口（样内-样外） | {_fmt_pct(wf_summary['return_gap'])} |",
            f"| 夏普缺口（样内-样外） | {_fmt_num(wf_summary['sharpe_gap'])} |",
            "",
            "### 4.1 每折表现预览",
            "",
            _table_markdown(wf_preview, max_rows=10),
            "",
        ]
    )

    if wf_equity_png.exists():
        lines.extend(
            [
                "### 4.2 样本外折间复利权益曲线",
                "",
                f"![wf_equity]({wf_equity_png.name})",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. 结论",
            "",
            f"- 综合判断：**{overfit_note}**。",
            "- 若样本外表现弱于样本内，建议缩小参数自由度、延长测试区间、增加跨品种验证。",
            "- 建议加入实盘约束（更严格滑点、手续费、流动性过滤）再次验证稳健性。",
            "",
            "> 本文档仅用于研究，不构成投资建议。",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    convert_md_to_pdf(md_path, pdf_path)

    print(f"文档Markdown: {md_path}")
    print(f"文档PDF: {pdf_path}")


if __name__ == "__main__":
    main()
