import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_system import fetch_bars, optimize_parameters, run_single_case


def generate_folds(
    n_bars: int,
    train_size: int,
    test_size: int,
    step_size: int,
    expanding: bool,
) -> list[tuple[int, int, int, int]]:
    folds = []
    start = 0
    while True:
        if expanding:
            train_start = 0
            train_end = train_size + start
        else:
            train_start = start
            train_end = train_start + train_size

        test_start = train_end
        test_end = test_start + test_size
        if test_end > n_bars:
            break

        folds.append((train_start, train_end, test_start, test_end))
        start += step_size
    return folds


def calc_fold_dd(equity_series: pd.Series) -> float:
    running_max = equity_series.cummax()
    dd = equity_series / running_max - 1
    return float(dd.min()) if len(dd) else 0.0


def summarize_results(fold_df: pd.DataFrame, initial_capital: float) -> dict:
    if fold_df.empty:
        return {
            "folds": 0,
            "oos_positive_ratio": 0.0,
            "oos_mean_return": 0.0,
            "oos_median_return": 0.0,
            "is_mean_return": 0.0,
            "is_mean_sharpe": 0.0,
            "oos_mean_sharpe": 0.0,
            "return_gap": 0.0,
            "sharpe_gap": 0.0,
            "oos_total_return_compounded": 0.0,
            "oos_max_drawdown_on_fold_equity": 0.0,
        }

    oos_compound_equity = [initial_capital]
    for r in fold_df["oos_total_return"]:
        oos_compound_equity.append(oos_compound_equity[-1] * (1 + float(r)))
    oos_compound_equity = pd.Series(oos_compound_equity)

    return {
        "folds": int(len(fold_df)),
        "oos_positive_ratio": float((fold_df["oos_total_return"] > 0).mean()),
        "oos_mean_return": float(fold_df["oos_total_return"].mean()),
        "oos_median_return": float(fold_df["oos_total_return"].median()),
        "is_mean_return": float(fold_df["is_total_return"].mean()),
        "is_mean_sharpe": float(fold_df["is_sharpe"].mean()),
        "oos_mean_sharpe": float(fold_df["oos_sharpe"].mean()),
        "return_gap": float(fold_df["is_total_return"].mean() - fold_df["oos_total_return"].mean()),
        "sharpe_gap": float(fold_df["is_sharpe"].mean() - fold_df["oos_sharpe"].mean()),
        "oos_total_return_compounded": float(oos_compound_equity.iloc[-1] / initial_capital - 1),
        "oos_max_drawdown_on_fold_equity": calc_fold_dd(oos_compound_equity),
    }


def plot_oos_fold_equity(
    fold_df: pd.DataFrame,
    initial_capital: float,
    out_png: Path,
) -> None:
    if fold_df.empty:
        return
    equity = [initial_capital]
    x = [0]
    for i, r in enumerate(fold_df["oos_total_return"], start=1):
        equity.append(equity[-1] * (1 + float(r)))
        x.append(i)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(x, equity, marker="o", color="#1f77b4")
    ax.set_title("Walk-Forward 样本外折间复利权益曲线")
    ax.set_xlabel("Fold 序号")
    ax.set_ylabel("权益")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    out_md: Path,
    symbol: str,
    strategy: str,
    fold_df: pd.DataFrame,
    summary: dict,
    fold_csv_name: str,
    chart_name: str,
) -> None:
    preview = fold_df.head(10).copy()
    headers = list(preview.columns)
    md_rows = []
    md_rows.append("| " + " | ".join(headers) + " |")
    md_rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in preview.iterrows():
        vals = [str(row[h]) for h in headers]
        md_rows.append("| " + " | ".join(vals) + " |")

    lines = [
        f"# Walk-Forward 样本外验证报告（{strategy}）",
        "",
        f"- 标的：`{symbol}`",
        f"- 折数：`{summary['folds']}`",
        "",
        "## 总结指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 样本外胜率（正收益折占比） | {summary['oos_positive_ratio']:.2%} |",
        f"| 样本外平均收益率（每折） | {summary['oos_mean_return']:.2%} |",
        f"| 样本外中位收益率（每折） | {summary['oos_median_return']:.2%} |",
        f"| 样本内平均收益率（每折） | {summary['is_mean_return']:.2%} |",
        f"| 样本内平均夏普 | {summary['is_mean_sharpe']:.3f} |",
        f"| 样本外平均夏普 | {summary['oos_mean_sharpe']:.3f} |",
        f"| 收益缺口（样内-样外） | {summary['return_gap']:.2%} |",
        f"| 夏普缺口（样内-样外） | {summary['sharpe_gap']:.3f} |",
        f"| 样本外复利总收益 | {summary['oos_total_return_compounded']:.2%} |",
        f"| 折间权益最大回撤 | {summary['oos_max_drawdown_on_fold_equity']:.2%} |",
        "",
        "## 过拟合判断参考",
        "",
        "- 若样内显著好于样外（收益缺口/夏普缺口过大），可能存在过拟合。",
        "- 若样外正收益折占比偏低（例如低于50%），策略鲁棒性偏弱。",
        "- 建议结合不同市场阶段、不同品种、不同交易成本再做稳健性验证。",
        "",
        "## 输出文件",
        "",
        f"- 每折结果：`{fold_csv_name}`",
        f"- 折间权益图：`{chart_name}`",
        "",
        f"![wf_oos_equity]({chart_name})",
        "",
        "## 每折结果预览（前10行）",
        "",
        *md_rows,
        "",
        "> 仅供研究，不构成投资建议。",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward out-of-sample validation with rolling optimization")
    parser.add_argument("--strategy", choices=["ma", "breakout"], default="ma")
    parser.add_argument("--symbol", default="KQ.m@INE.sc")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--bars", type=int, default=2600)
    parser.add_argument("--user", default=os.getenv("TQ_USER"))
    parser.add_argument("--password", default=os.getenv("TQ_PASSWORD"))

    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=160)
    parser.add_argument("--step-size", type=int, default=160)
    parser.add_argument("--expanding", action="store_true", help="Use expanding training window")

    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-contracts", type=int, default=20)
    parser.add_argument("--margin-rate", type=float, default=0.12)
    parser.add_argument("--max-margin-usage", type=float, default=0.5)
    parser.add_argument("--commission", type=float, default=20)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    parser.add_argument("--stop-atr", type=float, default=2.0)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--fast-ma", type=int, default=20)
    parser.add_argument("--slow-ma", type=int, default=60)
    parser.add_argument("--breakout-window", type=int, default=20)
    parser.add_argument("--atr-window", type=int, default=14)

    parser.add_argument("--grid-fast-ma", default="10,20,30")
    parser.add_argument("--grid-slow-ma", default="40,60,80")
    parser.add_argument("--grid-breakout-window", default="10,20,30")
    parser.add_argument("--grid-stop-atr", default="1.5,2.0,2.5")
    parser.add_argument("--grid-rr", default="1.5,2.0,2.5")
    parser.add_argument("--grid-risk", default="0.005,0.01,0.015")
    parser.add_argument("--min-trades-filter", type=int, default=15)
    parser.add_argument("--max-dd-filter", type=float, default=0.35)

    parser.add_argument("--output-dir", default="strategy_system_tool/output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.user or not args.password:
        raise SystemExit("Please provide --user/--password or env TQ_USER/TQ_PASSWORD")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_df, price_tick, vol_mult = fetch_bars(
        user=args.user,
        password=args.password,
        symbol=args.symbol,
        duration_seconds=args.duration,
        data_length=args.bars,
    )
    folds = generate_folds(
        n_bars=len(bars_df),
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        expanding=args.expanding,
    )
    if not folds:
        raise SystemExit("No valid folds. Increase bars or reduce train/test/step size.")

    rows = []
    for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(folds, start=1):
        train_df = bars_df.iloc[tr_s:tr_e].reset_index(drop=True)
        test_df = bars_df.iloc[te_s:te_e].reset_index(drop=True)

        _, best_params, _, _, _, is_metrics = optimize_parameters(
            bars_df=train_df,
            strategy=args.strategy,
            price_tick=price_tick,
            vol_mult=vol_mult,
            args=args,
        )
        _, _, _, oos_metrics = run_single_case(
            bars_df=test_df,
            strategy=args.strategy,
            params=best_params,
            price_tick=price_tick,
            vol_mult=vol_mult,
            args=args,
        )

        rows.append(
            {
                "fold_id": fold_id,
                "train_start": str(train_df["dt"].iloc[0]),
                "train_end": str(train_df["dt"].iloc[-1]),
                "test_start": str(test_df["dt"].iloc[0]),
                "test_end": str(test_df["dt"].iloc[-1]),
                "best_params": str(best_params),
                "is_trades": is_metrics["trades"],
                "is_win_rate": is_metrics["win_rate"],
                "is_total_return": is_metrics["total_return"],
                "is_sharpe": is_metrics["sharpe"],
                "is_max_dd": is_metrics["max_drawdown"],
                "oos_trades": oos_metrics["trades"],
                "oos_win_rate": oos_metrics["win_rate"],
                "oos_total_return": oos_metrics["total_return"],
                "oos_sharpe": oos_metrics["sharpe"],
                "oos_max_dd": oos_metrics["max_drawdown"],
            }
        )
        print(
            f"fold {fold_id}: IS ret {is_metrics['total_return']:.2%}, OOS ret {oos_metrics['total_return']:.2%}, "
            f"IS sharpe {is_metrics['sharpe']:.3f}, OOS sharpe {oos_metrics['sharpe']:.3f}"
        )

    fold_df = pd.DataFrame(rows)
    summary = summarize_results(fold_df, initial_capital=args.initial_capital)

    stem = f"wf_{args.strategy}_{args.symbol.replace('@', '_').replace('.', '_')}"
    fold_csv = out_dir / f"{stem}_folds.csv"
    summary_md = out_dir / f"{stem}_summary.md"
    equity_png = out_dir / f"{stem}_oos_equity.png"
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    plot_oos_fold_equity(fold_df, initial_capital=args.initial_capital, out_png=equity_png)
    write_summary_md(
        out_md=summary_md,
        symbol=args.symbol,
        strategy=args.strategy,
        fold_df=fold_df,
        summary=summary,
        fold_csv_name=fold_csv.name,
        chart_name=equity_png.name,
    )

    print(f"Walk-forward folds: {fold_csv}")
    print(f"Walk-forward summary: {summary_md}")
    print(
        f"OOS mean return: {summary['oos_mean_return']:.2%}, OOS mean sharpe: {summary['oos_mean_sharpe']:.3f}, "
        f"gaps -> return: {summary['return_gap']:.2%}, sharpe: {summary['sharpe_gap']:.3f}"
    )


if __name__ == "__main__":
    main()
