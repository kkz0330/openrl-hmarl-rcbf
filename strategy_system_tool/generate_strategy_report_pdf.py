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
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.90])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.0, 0.94, subtitle, fontsize=11, color="#666", va="top")
    return fig, ax


def add_bullets(ax, y_start: float, lines: list[str], fontsize: int = 12, line_gap: float = 0.048) -> float:
    y = y_start
    for line in lines:
        ax.text(0.0, y, f"- {line}", fontsize=fontsize, va="top")
        y -= line_gap
    return y


def draw_table(
    ax,
    columns: list[str],
    rows: list[list[Any]],
    bbox: tuple[float, float, float, float],
    fontsize: int = 10,
):
    tbl = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if r == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold")
    return tbl


def add_image(fig, image_path: Path, rect: tuple[float, float, float, float], title: str):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if not image_path.exists():
        ax.text(0.5, 0.5, f"未找到图片:\n{image_path.name}", ha="center", va="center", color="#B91C1C", fontsize=11)
        return
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_base_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {
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
        out["win_rate"] = float((pnl > 0).mean()) if len(pnl) > 0 else np.nan
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
    oos_sharpe = pd.to_numeric(wf["oos_sharpe"], errors="coerce").fillna(0.0)
    is_sharpe = pd.to_numeric(wf["is_sharpe"], errors="coerce").fillna(0.0)

    out["folds"] = float(len(wf))
    out["oos_positive_ratio"] = float((oos_ret > 0).mean())
    out["oos_mean_return"] = float(oos_ret.mean())
    out["oos_median_return"] = float(oos_ret.median())
    out["is_mean_return"] = float(is_ret.mean())
    out["return_gap"] = float(is_ret.mean() - oos_ret.mean())
    out["is_mean_sharpe"] = float(is_sharpe.mean())
    out["oos_mean_sharpe"] = float(oos_sharpe.mean())
    out["sharpe_gap"] = float(is_sharpe.mean() - oos_sharpe.mean())
    return out


def pct(v: float) -> str:
    return "N/A" if pd.isna(v) else f"{v * 100:.2f}%"


def num(v: float, n: int = 3) -> str:
    return "N/A" if pd.isna(v) else f"{v:.{n}f}"


def build_report(output_dir: Path, stem: str, output_pdf: Path) -> Path:
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

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        # Page 1
        fig, ax = new_page("交易策略说明文档", "strategy_system_tool 模块结果汇总")
        add_bullets(
            ax,
            0.86,
            [
                "本报告说明 strategy_system_tool 的策略逻辑、回测框架和验证结果。",
                "策略支持：均线策略（ma）与突破策略（breakout）。",
                "本次展示使用标的：KQ.m@INE.sc（主连），以 ma 策略结果为例。",
            ],
            fontsize=12,
            line_gap=0.05,
        )
        ax.text(0.0, 0.70, f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11, color="#444")
        add_image(fig, base_chart, (0.08, 0.12, 0.84, 0.48), "基础回测图（价格 + 权益）")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2
        fig, ax = new_page("1. 策略与风控逻辑", "来自 backtest_system.py")
        add_bullets(
            ax,
            0.88,
            [
                "入场信号：",
                "ma：快均线 > 慢均线做多，快均线 < 慢均线做空。",
                "breakout：上破前高做多，下破前低做空。",
                "出场条件：信号反转、ATR 止损、RR 止盈、回测结束强平。",
                "仓位控制：按风险预算和保证金双重约束取最小可开手数。",
                "风险约束：单笔风险比例、最大保证金占用、最大手数、手续费与滑点。",
            ],
            fontsize=11,
            line_gap=0.052,
        )
        add_bullets(
            ax,
            0.45,
            [
                "核心参数：fast_ma, slow_ma, breakout_window, atr_window, stop_atr, rr, risk_per_trade。",
                "支持网格优化：对参数组合批量回测并按评分排序。",
                "支持 Walk-Forward：每折先优化后测试，降低过拟合风险。",
            ],
            fontsize=11,
            line_gap=0.052,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3
        fig, ax = new_page("2. 基础回测结果", f"结果前缀：{stem}")
        rows = [
            ["交易次数", f"{int(base['trades'])}"],
            ["胜率", pct(base["win_rate"])],
            ["盈亏比", num(base["payoff_ratio"])],
            ["Profit Factor", num(base["profit_factor"])],
            ["总收益率", pct(base["total_return"])],
            ["最大回撤", pct(base["max_drawdown"])],
        ]
        draw_table(ax, ["指标", "结果"], rows, bbox=(0.00, 0.52, 0.50, 0.34), fontsize=12)

        if not trades.empty:
            trade_preview = trades.copy()
            trade_preview["entry_dt"] = trade_preview["entry_dt"].astype(str).str[:19]
            trade_preview["exit_dt"] = trade_preview["exit_dt"].astype(str).str[:19]
            preview = trade_preview[["entry_dt", "exit_dt", "direction", "qty", "pnl", "exit_reason"]].head(8)
            preview_rows = []
            for _, r in preview.iterrows():
                preview_rows.append(
                    [
                        r["entry_dt"],
                        r["exit_dt"],
                        str(r["direction"]),
                        str(int(r["qty"])),
                        f"{float(r['pnl']):.1f}",
                        str(r["exit_reason"]),
                    ]
                )
            draw_table(
                ax,
                ["开仓", "平仓", "方向", "手数", "盈亏", "原因"],
                preview_rows,
                bbox=(0.00, 0.08, 1.00, 0.36),
                fontsize=9,
            )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4
        fig, ax = new_page("3. 网格优化结果", "参数搜索与最优组合")
        if not grid.empty:
            best_rows = [
                ["fast_ma", str(int(best["fast_ma"])) if "fast_ma" in best else "N/A"],
                ["slow_ma", str(int(best["slow_ma"])) if "slow_ma" in best else "N/A"],
                ["stop_atr", num(float(best["stop_atr"])) if "stop_atr" in best else "N/A"],
                ["rr", num(float(best["rr"])) if "rr" in best else "N/A"],
                ["risk_per_trade", num(float(best["risk_per_trade"])) if "risk_per_trade" in best else "N/A"],
                ["score", num(float(best["score"])) if "score" in best else "N/A"],
                ["trades", str(int(best["trades"])) if "trades" in best else "N/A"],
                ["win_rate", pct(float(best["win_rate"])) if "win_rate" in best else "N/A"],
                ["total_return", pct(float(best["total_return"])) if "total_return" in best else "N/A"],
                ["max_drawdown", pct(float(best["max_drawdown"])) if "max_drawdown" in best else "N/A"],
                ["sharpe", num(float(best["sharpe"])) if "sharpe" in best else "N/A"],
            ]
            draw_table(ax, ["最优参数项", "值"], best_rows, bbox=(0.00, 0.52, 0.52, 0.38), fontsize=10)

            top = grid.sort_values("score", ascending=False).head(8).copy()
            top_rows = []
            for _, r in top.iterrows():
                top_rows.append(
                    [
                        str(int(r["case_id"])) if "case_id" in top.columns else "-",
                        num(float(r["score"])),
                        str(int(r["trades"])),
                        pct(float(r["win_rate"])),
                        pct(float(r["total_return"])),
                        pct(float(r["max_drawdown"])),
                        num(float(r["sharpe"])),
                    ]
                )
            draw_table(
                ax,
                ["case", "score", "trades", "win_rate", "return", "max_dd", "sharpe"],
                top_rows,
                bbox=(0.00, 0.08, 1.00, 0.36),
                fontsize=9,
            )
        else:
            ax.text(0.0, 0.82, "未找到网格优化结果文件。", fontsize=12, color="#B91C1C")
        add_image(fig, best_chart, (0.56, 0.52, 0.40, 0.38), "最优参数回测图")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5
        fig, ax = new_page("4. 样本外验证（Walk-Forward）", "稳健性与过拟合检查")
        wf_rows = [
            ["折数", str(int(wfm["folds"])) if not pd.isna(wfm["folds"]) else "N/A"],
            ["样本外正收益折占比", pct(wfm["oos_positive_ratio"])],
            ["样本外平均收益率(每折)", pct(wfm["oos_mean_return"])],
            ["样本外中位收益率(每折)", pct(wfm["oos_median_return"])],
            ["样本内平均收益率(每折)", pct(wfm["is_mean_return"])],
            ["收益缺口(样内-样外)", pct(wfm["return_gap"])],
            ["样本内平均夏普", num(wfm["is_mean_sharpe"])],
            ["样本外平均夏普", num(wfm["oos_mean_sharpe"])],
            ["夏普缺口(样内-样外)", num(wfm["sharpe_gap"])],
        ]
        draw_table(ax, ["指标", "结果"], wf_rows, bbox=(0.00, 0.56, 0.52, 0.34), fontsize=10)
        add_image(fig, wf_chart, (0.56, 0.56, 0.40, 0.34), "样本外折间权益曲线")

        overfit_hint = "存在过拟合风险"
        if not pd.isna(wfm["oos_positive_ratio"]) and wfm["oos_positive_ratio"] >= 0.5 and not pd.isna(wfm["return_gap"]) and wfm["return_gap"] <= 0.005:
            overfit_hint = "过拟合风险可控"
        add_bullets(
            ax,
            0.45,
            [
                f"结论：{overfit_hint}（基于收益缺口、夏普缺口、正收益折占比综合判断）。",
                "建议增加跨市场、跨周期、跨参数密度的稳定性测试。",
                "建议加入更严格交易成本和滑点模型后再评估可实盘性。",
            ],
            fontsize=11,
            line_gap=0.055,
        )
        pdf.savefig(fig)
        plt.close(fig)

    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 strategy_system_tool 的交易策略说明 PDF")
    parser.add_argument("--output-dir", default="strategy_system_tool/output", help="结果目录")
    parser.add_argument("--stem", default="ma_KQ_m_INE_sc", help="结果文件前缀")
    parser.add_argument("--output-pdf", default="", help="输出 PDF 路径")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if args.output_pdf:
        output_pdf = Path(args.output_pdf).resolve()
    else:
        output_pdf = output_dir / f"{args.stem}_strategy_overview.pdf"

    configure_fonts()
    result = build_report(output_dir=output_dir, stem=args.stem, output_pdf=output_pdf)
    print(f"PDF 已生成: {result}")


if __name__ == "__main__":
    main()
