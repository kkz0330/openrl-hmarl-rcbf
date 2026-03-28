from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from backtest_system import build_parser


def configure_fonts() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    selected = None
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            selected = name
            break
        except Exception:
            continue
    plt.rcParams["font.family"] = selected if selected else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def new_page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.91])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.0, 0.94, subtitle, fontsize=11, color="#555", va="top")
    return fig, ax


def add_bullets(ax, y_start: float, lines: list[str], fontsize: int = 11, line_gap: float = 0.046):
    y = y_start
    for line in lines:
        ax.text(0.0, y, f"- {line}", fontsize=fontsize, va="top")
        y -= line_gap
    return y


def draw_table(ax, columns: list[str], rows: list[list[Any]], bbox: tuple[float, float, float, float], fontsize: int = 10):
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if r == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold")
    return table


def add_image(fig, path: Path, rect: tuple[float, float, float, float], title: str):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if not path.exists():
        ax.text(0.5, 0.5, f"未找到图片:\n{path.name}", ha="center", va="center", color="#B91C1C")
        return
    img = plt.imread(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(v: float) -> str:
    return "N/A" if pd.isna(v) else f"{v * 100:.2f}%"


def num(v: float, n: int = 3) -> str:
    return "N/A" if pd.isna(v) else f"{v:.{n}f}"


def compute_base_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    out = {
        "trades": 0.0,
        "win_rate": np.nan,
        "payoff_ratio": np.nan,
        "profit_factor": np.nan,
        "total_return": np.nan,
        "max_drawdown": np.nan,
    }
    if not trades.empty:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        out["trades"] = float(len(pnl))
        out["win_rate"] = float((pnl > 0).mean())
        avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else np.nan
        avg_loss = -pnl[pnl < 0].mean() if (pnl < 0).any() else np.nan
        out["payoff_ratio"] = float(avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        out["profit_factor"] = float(wins / losses) if losses > 0 else np.nan

    if not equity.empty and "equity" in equity.columns:
        eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if len(eq) > 1:
            out["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            out["max_drawdown"] = float((eq / eq.cummax() - 1.0).min())
    return out


def compute_wf_metrics(wf: pd.DataFrame) -> dict[str, float]:
    out = {
        "folds": 0.0,
        "oos_positive_ratio": np.nan,
        "oos_mean_return": np.nan,
        "oos_median_return": np.nan,
        "is_mean_return": np.nan,
        "return_gap": np.nan,
        "is_mean_sharpe": np.nan,
        "oos_mean_sharpe": np.nan,
        "sharpe_gap": np.nan,
    }
    if wf.empty:
        return out

    oos_ret = pd.to_numeric(wf["oos_total_return"], errors="coerce").fillna(0.0)
    is_ret = pd.to_numeric(wf["is_total_return"], errors="coerce").fillna(0.0)
    oos_sh = pd.to_numeric(wf["oos_sharpe"], errors="coerce").fillna(0.0)
    is_sh = pd.to_numeric(wf["is_sharpe"], errors="coerce").fillna(0.0)

    out["folds"] = float(len(wf))
    out["oos_positive_ratio"] = float((oos_ret > 0).mean())
    out["oos_mean_return"] = float(oos_ret.mean())
    out["oos_median_return"] = float(oos_ret.median())
    out["is_mean_return"] = float(is_ret.mean())
    out["return_gap"] = float(is_ret.mean() - oos_ret.mean())
    out["is_mean_sharpe"] = float(is_sh.mean())
    out["oos_mean_sharpe"] = float(oos_sh.mean())
    out["sharpe_gap"] = float(is_sh.mean() - oos_sh.mean())
    return out


def draw_state_machine(fig, rect: tuple[float, float, float, float]):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "#EEF2FF"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#4B5563", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.5, 7.2, 2.5, 1.2, "空仓\nFlat")
    box(3.8, 7.2, 2.5, 1.2, "多头持仓\nLong")
    box(7.0, 7.2, 2.5, 1.2, "空头持仓\nShort")
    box(0.5, 4.4, 2.5, 1.2, "风控闸门\nRisk Gate", fc="#ECFDF5")
    box(3.8, 4.4, 2.5, 1.2, "平仓执行\nClose", fc="#FEF3C7")
    box(7.0, 4.4, 2.5, 1.2, "权益更新\nEquity", fc="#F3F4F6")

    arrow = dict(arrowstyle="->", color="#374151", lw=1.2)
    ax.annotate("", xy=(1.8, 6.8), xytext=(1.8, 5.6), arrowprops=arrow)
    ax.text(2.0, 6.2, "有信号", fontsize=9)
    ax.annotate("", xy=(3.8, 7.8), xytext=(3.0, 7.8), arrowprops=arrow)
    ax.annotate("", xy=(7.0, 7.8), xytext=(3.0, 7.8), arrowprops=arrow)
    ax.text(4.1, 8.2, "通过仓位/保证金检查", fontsize=9)
    ax.annotate("", xy=(5.0, 6.8), xytext=(5.0, 5.6), arrowprops=arrow)
    ax.annotate("", xy=(8.2, 6.8), xytext=(5.0, 5.6), arrowprops=arrow)
    ax.text(5.2, 6.2, "反向/止损/止盈/结束", fontsize=9)
    ax.annotate("", xy=(8.2, 5.6), xytext=(8.2, 4.0), arrowprops=arrow)
    ax.annotate("", xy=(1.8, 7.2), xytext=(8.2, 4.0), arrowprops=arrow)
    ax.text(5.7, 4.6, "更新现金与浮盈亏", fontsize=9)


def build_enhanced_report(output_dir: Path, stem: str, output_pdf: Path) -> Path:
    trades = safe_read_csv(output_dir / f"{stem}_trades.csv")
    equity = safe_read_csv(output_dir / f"{stem}_equity.csv")
    grid = safe_read_csv(output_dir / f"{stem}_grid_compare.csv")
    wf = safe_read_csv(output_dir / f"wf_{stem}_folds.csv")

    base = compute_base_metrics(trades, equity)
    wfm = compute_wf_metrics(wf)
    best = grid.sort_values("score", ascending=False).iloc[0] if not grid.empty else pd.Series(dtype=float)

    base_chart = output_dir / f"{stem}_chart.png"
    best_chart = output_dir / f"{stem}_best_chart.png"
    wf_chart = output_dir / f"wf_{stem}_oos_equity.png"

    parser = build_parser()
    defaults = vars(parser.parse_args([]))

    exit_reason_counts = {}
    if not trades.empty and "exit_reason" in trades.columns:
        exit_reason_counts = trades["exit_reason"].value_counts().to_dict()

    overfit_flag = "存在过拟合风险"
    if (
        not pd.isna(wfm["oos_positive_ratio"])
        and wfm["oos_positive_ratio"] >= 0.5
        and not pd.isna(wfm["return_gap"])
        and wfm["return_gap"] <= 0.005
        and not pd.isna(wfm["sharpe_gap"])
        and wfm["sharpe_gap"] <= 1.0
    ):
        overfit_flag = "过拟合风险可控"

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        # Page 1: cover + summary
        fig, ax = new_page("交易系统增强版说明文档", f"模块: strategy_system_tool | 前缀: {stem}")
        add_bullets(
            ax,
            0.88,
            [
                "本报告补充完整交易系统要素：开仓、平仓、资金管理、风控与异常处理。",
                "结果基于当前 output 目录中的回测、网格优化、Walk-Forward 文件自动生成。",
                "重点用于策略研发评审与工程交接，不作为投资建议。",
            ],
        )
        summary_rows = [
            ["交易次数", f"{int(base['trades'])}"],
            ["基础回测收益率", pct(base["total_return"])],
            ["基础回测最大回撤", pct(base["max_drawdown"])],
            ["网格最优收益率", pct(float(best.get("total_return", np.nan)))],
            ["网格最优夏普", num(float(best.get("sharpe", np.nan)))],
            ["样本外正收益折占比", pct(wfm["oos_positive_ratio"])],
            ["过拟合判断", overfit_flag],
        ]
        draw_table(ax, ["摘要指标", "结果"], summary_rows, bbox=(0.00, 0.50, 0.56, 0.34), fontsize=11)
        add_image(fig, base_chart, (0.58, 0.50, 0.40, 0.34), "基础回测图")
        ax.text(0.0, 0.46, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10, color="#666")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: architecture + state machine
        fig, ax = new_page("1. 交易系统架构与状态机", "从信号到下单到风控的闭环")
        add_bullets(
            ax,
            0.90,
            [
                "数据层：TqSdk 获取K线/行情 -> 计算指标（MA、ATR、突破上下沿）。",
                "决策层：desired_signal 生成方向信号；run_backtest 驱动持仓状态转移。",
                "执行层：按开盘价+滑点成交，写入 trade ledger 和 equity 曲线。",
                "控制层：仓位计算 + 保证金约束 + 止损止盈 + 手续费。",
            ],
            fontsize=10,
            line_gap=0.044,
        )
        draw_state_machine(fig, (0.06, 0.10, 0.88, 0.48))
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: rules + capital management formulas
        fig, ax = new_page("2. 开平仓与资金管理规则", "规则表 + 公式")
        rule_rows = [
            ["开仓", "MA策略", "fast_ma > slow_ma 做多；fast_ma < slow_ma 做空"],
            ["开仓", "突破策略", "close > 前高做多；close < 前低做空"],
            ["平仓", "信号反转", "持多遇空信号/持空遇多信号，按下一根开盘价反手平仓"],
            ["平仓", "止损", "long: low <= stop_price；short: high >= stop_price"],
            ["平仓", "止盈", "long: high >= take_price；short: low <= take_price"],
            ["平仓", "强平", "回测最后一根K线按收盘价强制平仓"],
        ]
        draw_table(ax, ["模块", "条件类型", "实现规则"], rule_rows, bbox=(0.00, 0.50, 1.00, 0.40), fontsize=9)

        formula_lines = [
            "风险预算: risk_budget = equity * risk_per_trade",
            "止损距离: stop_distance = ATR * stop_atr",
            "风险手数: qty_by_risk = floor(risk_budget / (stop_distance * volume_multiple))",
            "保证金手数: qty_by_margin = floor((equity * max_margin_usage) / (entry_price * volume_multiple * margin_rate))",
            "最终手数: qty = max(0, min(qty_by_risk, qty_by_margin, max_contracts))",
            "止盈距离: take_distance = stop_distance * rr",
            "成交价格(含滑点): long_entry=open+slippage, short_entry=open-slippage",
        ]
        y = add_bullets(ax, 0.44, formula_lines, fontsize=10, line_gap=0.042)
        ax.text(0.0, y - 0.01, "备注：slippage = slippage_ticks * price_tick", fontsize=10, color="#555")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: parameter dictionary and defaults vs best
        fig, ax = new_page("3. 参数字典与配置基线", "默认参数 vs 当前网格最优参数")
        default_rows = [
            ["initial_capital", defaults["initial_capital"], "初始资金"],
            ["risk_per_trade", defaults["risk_per_trade"], "单笔风险比例"],
            ["max_contracts", defaults["max_contracts"], "最大开仓手数"],
            ["margin_rate", defaults["margin_rate"], "保证金率"],
            ["max_margin_usage", defaults["max_margin_usage"], "最大保证金占比"],
            ["commission", defaults["commission"], "单手手续费"],
            ["slippage_ticks", defaults["slippage_ticks"], "滑点tick数"],
            ["stop_atr", defaults["stop_atr"], "止损ATR倍数"],
            ["rr", defaults["rr"], "止盈风险收益比"],
            ["fast_ma", defaults["fast_ma"], "快均线周期"],
            ["slow_ma", defaults["slow_ma"], "慢均线周期"],
            ["atr_window", defaults["atr_window"], "ATR窗口"],
        ]
        draw_table(ax, ["参数", "默认值", "含义"], default_rows, bbox=(0.00, 0.50, 0.58, 0.42), fontsize=9)

        if not grid.empty:
            best_rows = [
                ["fast_ma", int(best["fast_ma"]) if "fast_ma" in best else "N/A"],
                ["slow_ma", int(best["slow_ma"]) if "slow_ma" in best else "N/A"],
                ["stop_atr", num(float(best["stop_atr"])) if "stop_atr" in best else "N/A"],
                ["rr", num(float(best["rr"])) if "rr" in best else "N/A"],
                ["risk_per_trade", num(float(best["risk_per_trade"])) if "risk_per_trade" in best else "N/A"],
                ["score", num(float(best["score"]))],
                ["trades", int(best["trades"])],
                ["win_rate", pct(float(best["win_rate"]))],
                ["total_return", pct(float(best["total_return"]))],
                ["max_drawdown", pct(float(best["max_drawdown"]))],
                ["sharpe", num(float(best["sharpe"]))],
            ]
            draw_table(ax, ["最优项", "值"], best_rows, bbox=(0.62, 0.50, 0.38, 0.42), fontsize=9)
        add_image(fig, best_chart, (0.00, 0.08, 0.48, 0.34), "最优参数回测图")
        add_image(fig, wf_chart, (0.52, 0.08, 0.48, 0.34), "样本外权益图")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: risk matrix + boundary / exception handling
        fig, ax = new_page("4. 风控矩阵与边界条件", "代码层防护清单")
        risk_rows = [
            ["单笔风险", "risk_per_trade + ATR定仓", "calc_position_size", "已实现"],
            ["保证金约束", "max_margin_usage", "calc_position_size", "已实现"],
            ["仓位上限", "max_contracts", "calc_position_size", "已实现"],
            ["价格冲击", "slippage_ticks", "run_backtest", "已实现"],
            ["交易成本", "commission", "run_backtest", "已实现"],
            ["止损", "ATR * stop_atr", "run_backtest", "已实现"],
            ["止盈", "stop_distance * rr", "run_backtest", "已实现"],
        ]
        draw_table(ax, ["风控项", "参数/规则", "函数", "状态"], risk_rows, bbox=(0.00, 0.54, 1.00, 0.34), fontsize=9)

        exception_rows = [
            ["ATR无效", "返回0手，不开仓", "calc_position_size"],
            ["price_tick缺失", "降级到0.1", "fetch_bars"],
            ["volume_multiple缺失", "降级到1.0", "fetch_bars"],
            ["网格为空", "抛出RuntimeError", "optimize_parameters"],
            ["账号缺失", "直接终止运行", "main"],
            ["样本结束仍有持仓", "force_close_end 强平", "run_backtest"],
        ]
        draw_table(ax, ["异常/边界", "处理机制", "函数"], exception_rows, bbox=(0.00, 0.12, 1.00, 0.34), fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: result interpretation and actions
        fig, ax = new_page("5. 结果解读与改进建议", "工程可执行建议")
        exit_rows = [[k, str(v)] for k, v in sorted(exit_reason_counts.items(), key=lambda x: -x[1])] if exit_reason_counts else [["N/A", "N/A"]]
        draw_table(ax, ["平仓原因", "次数"], exit_rows[:8], bbox=(0.00, 0.62, 0.35, 0.24), fontsize=10)

        interpretation = [
            f"基础回测：收益 {pct(base['total_return'])}，最大回撤 {pct(base['max_drawdown'])}，胜率 {pct(base['win_rate'])}。",
            f"网格优化最优：收益 {pct(float(best.get('total_return', np.nan)))}，夏普 {num(float(best.get('sharpe', np.nan)))}。",
            f"样本外验证：正收益折占比 {pct(wfm['oos_positive_ratio'])}，收益缺口 {pct(wfm['return_gap'])}，夏普缺口 {num(wfm['sharpe_gap'])}。",
            f"综合结论：{overfit_flag}。",
        ]
        y = add_bullets(ax, 0.56, interpretation, fontsize=11, line_gap=0.052)
        add_bullets(
            ax,
            y - 0.02,
            [
                "建议1：将网格搜索改为分层/贝叶斯搜索，减少参数暴露。",
                "建议2：增加交易时段过滤与流动性阈值，抑制异常成交。",
                "建议3：在Walk-Forward中扩大折数并引入交易成本压力测试。",
                "建议4：加入结构识别模块（升贴水/波动 regime）做参数切换。",
            ],
            fontsize=11,
            line_gap=0.052,
        )
        ax.text(0.0, 0.08, "注：本报告用于研发评审，不构成任何投资建议。", fontsize=10, color="#666")
        pdf.savefig(fig)
        plt.close(fig)

    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 strategy_system_tool 增强版交易系统说明 PDF")
    parser.add_argument("--output-dir", default="strategy_system_tool/output", help="结果目录")
    parser.add_argument("--stem", default="ma_KQ_m_INE_sc", help="结果文件前缀")
    parser.add_argument("--output-pdf", default="", help="输出PDF路径")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_pdf = Path(args.output_pdf).resolve() if args.output_pdf else output_dir / f"{args.stem}_strategy_overview_enhanced.pdf"

    configure_fonts()
    path = build_enhanced_report(output_dir=output_dir, stem=args.stem, output_pdf=output_pdf)
    print(f"PDF 已生成: {path}")


if __name__ == "__main__":
    main()
