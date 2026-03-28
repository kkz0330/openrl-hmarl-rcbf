from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = ROOT / "docs" / "project_overview_report.pdf"


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
        ax.text(0.0, 0.94, subtitle, fontsize=11, color="#666666", va="top")
    return fig, ax


def add_bullets(ax, y_start: float, lines: list[str], fontsize: int = 12, line_gap: float = 0.045) -> float:
    y = y_start
    for line in lines:
        ax.text(0.0, y, f"- {line}", fontsize=fontsize, va="top")
        y -= line_gap
    return y


def draw_table(ax, columns: list[str], rows: list[list[Any]], bbox: tuple[float, float, float, float], fontsize: int = 10):
    tbl = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if r == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold")
    return tbl


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_strategy_metrics() -> dict[str, Any]:
    out_dir = ROOT / "strategy_system_tool" / "output"
    trades = safe_read_csv(out_dir / "ma_KQ_m_INE_sc_trades.csv")
    equity = safe_read_csv(out_dir / "ma_KQ_m_INE_sc_equity.csv")
    grid = safe_read_csv(out_dir / "ma_KQ_m_INE_sc_grid_compare.csv")
    wf = safe_read_csv(out_dir / "wf_ma_KQ_m_INE_sc_folds.csv")

    result: dict[str, Any] = {
        "trades": len(trades),
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "payoff_ratio": np.nan,
        "total_return": np.nan,
        "max_drawdown": np.nan,
        "best_grid": {},
        "wf_positive_fold_ratio": np.nan,
        "wf_mean_oos_return": np.nan,
        "wf_mean_oos_sharpe": np.nan,
    }

    if not trades.empty:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        result["win_rate"] = float((pnl > 0).mean())
        result["profit_factor"] = float(wins / losses) if losses > 0 else np.nan
        avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else np.nan
        avg_loss = -pnl[pnl < 0].mean() if (pnl < 0).any() else np.nan
        result["payoff_ratio"] = float(avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan

    if not equity.empty and "equity" in equity.columns:
        eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if len(eq) >= 2:
            result["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            dd = (eq / eq.cummax() - 1.0).min()
            result["max_drawdown"] = float(dd)

    if not grid.empty:
        top = grid.sort_values("score", ascending=False).iloc[0]
        result["best_grid"] = {
            "fast_ma": int(top["fast_ma"]),
            "slow_ma": int(top["slow_ma"]),
            "stop_atr": float(top["stop_atr"]),
            "rr": float(top["rr"]),
            "risk_per_trade": float(top["risk_per_trade"]),
            "score": float(top["score"]),
            "return": float(top["total_return"]),
            "sharpe": float(top["sharpe"]),
            "max_dd": float(top["max_drawdown"]),
            "trades": int(top["trades"]),
            "win_rate": float(top["win_rate"]),
        }

    if not wf.empty:
        oos_ret = pd.to_numeric(wf["oos_total_return"], errors="coerce").fillna(0.0)
        oos_sharpe = pd.to_numeric(wf["oos_sharpe"], errors="coerce").fillna(0.0)
        result["wf_positive_fold_ratio"] = float((oos_ret > 0).mean())
        result["wf_mean_oos_return"] = float(oos_ret.mean())
        result["wf_mean_oos_sharpe"] = float(oos_sharpe.mean())

    return result


def compute_term_structure_metrics() -> dict[str, Any]:
    csv_path = ROOT / "term_structure_tool" / "output" / "INE_sc_term_structure.csv"
    df = safe_read_csv(csv_path)
    if df.empty:
        return {"available": False}

    df = df.copy()
    df["expire_dt"] = pd.to_datetime(df["expire_dt"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["expire_dt", "close"]).sort_values("expire_dt").reset_index(drop=True)
    if df.empty:
        return {"available": False}

    near = float(df["close"].iloc[0])
    far = float(df["close"].iloc[-1])
    spread = far - near
    spread_pct = spread / near if near else np.nan
    slope = float(np.polyfit(np.arange(len(df), dtype=float), df["close"].to_numpy(dtype=float), 1)[0]) if len(df) >= 2 else 0.0

    threshold = 0.01
    if spread_pct > threshold and slope > 0:
        structure = "升水结构（Contango）"
    elif spread_pct < -threshold and slope < 0:
        structure = "贴水结构（Backwardation）"
    else:
        structure = "平坦/混合结构"

    return {
        "available": True,
        "contracts": len(df),
        "near_symbol": str(df["symbol"].iloc[0]),
        "far_symbol": str(df["symbol"].iloc[-1]),
        "near_close": near,
        "far_close": far,
        "spread": spread,
        "spread_pct": spread_pct,
        "slope": slope,
        "structure": structure,
    }


def collect_project_snapshot() -> dict[str, Any]:
    module_paths = {
        "行情看板模块": ROOT / "ine_crude_dashboard.py",
        "期限结构模块": ROOT / "term_structure_tool",
        "策略回测模块": ROOT / "strategy_system_tool",
        "分层强化学习模块": ROOT / "hmarl_cbf",
        "测试集": ROOT / "tests",
    }

    rows = []
    for name, path in module_paths.items():
        if path.is_file():
            py_count = 1
            file_count = 1
        elif path.exists():
            py_count = len(list(path.rglob("*.py")))
            file_count = len([p for p in path.rglob("*") if p.is_file()])
        else:
            py_count = 0
            file_count = 0
        rows.append({"模块": name, "路径": str(path.relative_to(ROOT)), "Python文件数": py_count, "文件总数": file_count})

    return {"module_rows": rows}


def add_image(fig, img_path: Path, rect: tuple[float, float, float, float], title: str):
    if not img_path.exists():
        ax = fig.add_axes(rect)
        ax.axis("off")
        ax.text(0.5, 0.5, f"未找到图片:\n{img_path}", ha="center", va="center", color="#B91C1C", fontsize=11)
        return

    img = plt.imread(img_path)
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def build_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_fonts()

    snapshot = collect_project_snapshot()
    strategy = compute_strategy_metrics()
    term = compute_term_structure_metrics()

    dashboard_img = ROOT / "docs" / "dashboard_preview.png"
    strategy_best_img = ROOT / "strategy_system_tool" / "output" / "ma_KQ_m_INE_sc_best_chart.png"
    wf_img = ROOT / "strategy_system_tool" / "output" / "wf_ma_KQ_m_INE_sc_oos_equity.png"
    term_img = ROOT / "term_structure_tool" / "output" / "INE_sc_term_structure.png"

    with PdfPages(output_path) as pdf:
        # Page 1: cover
        fig, ax = new_page("项目说明文档", "自动汇总项目结构、关键结果与图表展示")
        y = add_bullets(
            ax,
            0.87,
            [
                "项目目标：构建面向原油量化研究的分析与回测工具集。",
                "主要能力：行情可视化、多模型预测、期限结构识别、策略回测与参数优化。",
                "结果来源：strategy_system_tool/output 与 term_structure_tool/output。",
            ],
            fontsize=12,
            line_gap=0.05,
        )
        ax.text(0.0, y - 0.02, f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11, color="#555")
        ax.text(0.0, y - 0.08, f"文档路径：{output_path}", fontsize=10, color="#333")
        add_image(fig, dashboard_img, (0.10, 0.12, 0.80, 0.38), "行情看板预览")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: project structure
        fig, ax = new_page("1. 项目结构总览", "模块划分与代码规模概览")
        rows = [[r["模块"], r["路径"], r["Python文件数"], r["文件总数"]] for r in snapshot["module_rows"]]
        draw_table(ax, ["模块", "路径", "Python文件数", "文件总数"], rows, bbox=(0.00, 0.45, 1.00, 0.43), fontsize=11)
        add_bullets(
            ax,
            0.36,
            [
                "行情看板模块：基于 TqSdk 读取主连与多合约数据，输出 HTML 可视化报告。",
                "期限结构模块：输出同品种多月合约收盘价曲线并判定升贴水。",
                "策略回测模块：提供单次回测、网格优化、滚动样本外验证。",
                "hmarl_cbf 模块：分层控制/强化学习相关实验代码与训练脚本。",
            ],
            fontsize=11,
            line_gap=0.05,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: strategy metrics
        fig, ax = new_page("2. 策略回测结果", "标的：KQ.m@INE.sc（均线策略）")
        metric_rows = [
            ["交易次数", f"{strategy['trades']}"],
            ["胜率", f"{strategy['win_rate'] * 100:.2f}%" if pd.notna(strategy["win_rate"]) else "N/A"],
            ["盈亏比", f"{strategy['payoff_ratio']:.3f}" if pd.notna(strategy["payoff_ratio"]) else "N/A"],
            ["Profit Factor", f"{strategy['profit_factor']:.3f}" if pd.notna(strategy["profit_factor"]) else "N/A"],
            ["总收益率", f"{strategy['total_return'] * 100:.2f}%" if pd.notna(strategy["total_return"]) else "N/A"],
            ["最大回撤", f"{strategy['max_drawdown'] * 100:.2f}%" if pd.notna(strategy["max_drawdown"]) else "N/A"],
            ["样本外正收益折占比", f"{strategy['wf_positive_fold_ratio'] * 100:.2f}%" if pd.notna(strategy["wf_positive_fold_ratio"]) else "N/A"],
            ["样本外平均收益率(每折)", f"{strategy['wf_mean_oos_return'] * 100:.2f}%" if pd.notna(strategy["wf_mean_oos_return"]) else "N/A"],
            ["样本外平均夏普", f"{strategy['wf_mean_oos_sharpe']:.3f}" if pd.notna(strategy["wf_mean_oos_sharpe"]) else "N/A"],
        ]
        draw_table(ax, ["指标", "数值"], metric_rows, bbox=(0.00, 0.44, 0.56, 0.46), fontsize=11)

        best = strategy.get("best_grid", {})
        best_rows = [
            ["fast_ma", best.get("fast_ma", "N/A")],
            ["slow_ma", best.get("slow_ma", "N/A")],
            ["stop_atr", best.get("stop_atr", "N/A")],
            ["rr", best.get("rr", "N/A")],
            ["risk_per_trade", best.get("risk_per_trade", "N/A")],
            ["score", f"{best.get('score', np.nan):.3f}" if best else "N/A"],
            ["网格最优收益率", f"{best.get('return', np.nan) * 100:.2f}%" if best else "N/A"],
            ["网格最优夏普", f"{best.get('sharpe', np.nan):.3f}" if best else "N/A"],
        ]
        draw_table(ax, ["最优参数项", "值"], best_rows, bbox=(0.60, 0.44, 0.40, 0.46), fontsize=10)
        add_bullets(
            ax,
            0.35,
            [
                "网格优化显示，最优参数组合获得正收益与较高估算夏普。",
                "滚动样本外验证均值仍偏弱，提示策略存在过拟合风险。",
                "建议优先做样本外稳健性改进，再考虑扩大资金规模。",
            ],
            fontsize=11,
            line_gap=0.05,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: strategy figures
        fig, ax = new_page("3. 策略图表展示", "最优回测曲线 + 样本外折间权益曲线")
        add_image(fig, strategy_best_img, (0.08, 0.50, 0.84, 0.36), "网格最优参数回测图")
        add_image(fig, wf_img, (0.08, 0.12, 0.84, 0.30), "滚动验证样本外权益曲线")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: term structure results
        fig, ax = new_page("4. 期限结构结果", "品种：INE.sc")
        if term.get("available"):
            term_rows = [
                ["合约数量", f"{term['contracts']}"],
                ["近月合约", term["near_symbol"]],
                ["远月合约", term["far_symbol"]],
                ["近月收盘价", f"{term['near_close']:.2f}"],
                ["远月收盘价", f"{term['far_close']:.2f}"],
                ["远近月价差", f"{term['spread']:.2f}"],
                ["远近月价差比例", f"{term['spread_pct'] * 100:.2f}%"],
                ["曲线斜率(线性拟合)", f"{term['slope']:.4f}"],
                ["结构判断", term["structure"]],
            ]
            draw_table(ax, ["指标", "结果"], term_rows, bbox=(0.00, 0.48, 0.52, 0.42), fontsize=11)
            add_bullets(
                ax,
                0.42,
                [
                    "若为贴水结构，可重点跟踪近强远弱是否继续。",
                    "若出现结构快速反转，需及时调整跨期价差策略方向。",
                    "实际交易需考虑流动性、换月、保证金与极端事件风险。",
                ],
                fontsize=10,
                line_gap=0.045,
            )
        else:
            ax.text(0.0, 0.80, "未检测到期限结构数据文件，无法展示该部分结果。", fontsize=12, color="#B91C1C")
        add_image(fig, term_img, (0.56, 0.12, 0.40, 0.76), "期限结构曲线")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: conclusion
        fig, ax = new_page("5. 结论与后续计划", "项目阶段性总结")
        add_bullets(
            ax,
            0.88,
            [
                "项目已形成“行情观察 -> 结构分析 -> 回测验证”的闭环工具链。",
                "已有结果显示：参数优化可显著改善样本内表现，但样本外稳定性仍需加强。",
                "期限结构当前表现为贴水（Backwardation）特征，可用于跨期研究。",
                "建议下一步：加入交易成本敏感性、结构切换检测和多市场对比验证。",
            ],
            fontsize=12,
            line_gap=0.055,
        )
        ax.text(0.0, 0.53, "附注：本报告用于研究与工程文档展示，不构成投资建议。", fontsize=11, color="#374151")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    build_pdf(OUTPUT_PDF)
    print(f"Generated: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
