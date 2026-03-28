import argparse
import itertools
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqsdk import TqApi, TqAuth


@dataclass
class Position:
    direction: int  # 1 long, -1 short
    qty: int
    entry_price: float
    entry_idx: int
    stop_price: float
    take_price: float


def safe_float(v: object) -> float:
    try:
        if v is None:
            return math.nan
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def wait_updates(api: TqApi, seconds: int = 25) -> None:
    end = time.time() + seconds
    while time.time() < end:
        api.wait_update(deadline=time.time() + 1)


def fetch_bars(
    user: str,
    password: str,
    symbol: str,
    duration_seconds: int,
    data_length: int,
) -> tuple[pd.DataFrame, float, float]:
    api = TqApi(auth=TqAuth(user, password))
    try:
        quote = api.get_quote(symbol)
        kl = api.get_kline_serial(symbol, duration_seconds, data_length=data_length)
        wait_updates(api, seconds=25)

        df = kl.copy()
        df["dt"] = pd.to_datetime(
            pd.to_numeric(df["datetime"], errors="coerce").fillna(0).astype("int64"),
            unit="ns",
            utc=True,
        ).dt.tz_convert("Asia/Shanghai")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        price_tick = safe_float(quote.price_tick)
        volume_multiple = safe_float(quote.volume_multiple)
        if not math.isfinite(price_tick) or price_tick <= 0:
            price_tick = 0.1
        if not math.isfinite(volume_multiple) or volume_multiple <= 0:
            volume_multiple = 1.0
        return df, price_tick, volume_multiple
    finally:
        api.close()


def add_indicators(
    df: pd.DataFrame,
    strategy: str,
    fast_ma: int,
    slow_ma: int,
    breakout_window: int,
    atr_window: int,
) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(atr_window).mean()

    if strategy == "ma":
        out["fast_ma"] = out["close"].rolling(fast_ma).mean()
        out["slow_ma"] = out["close"].rolling(slow_ma).mean()
    elif strategy == "breakout":
        out["hh"] = out["high"].rolling(breakout_window).max().shift(1)
        out["ll"] = out["low"].rolling(breakout_window).min().shift(1)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return out


def desired_signal(df: pd.DataFrame, i: int, strategy: str, current_pos: int) -> int:
    row = df.iloc[i]
    if strategy == "ma":
        fast = safe_float(row["fast_ma"])
        slow = safe_float(row["slow_ma"])
        if not math.isfinite(fast) or not math.isfinite(slow):
            return 0
        if fast > slow:
            return 1
        if fast < slow:
            return -1
        return 0

    close_price = safe_float(row["close"])
    hh = safe_float(row["hh"])
    ll = safe_float(row["ll"])
    if math.isfinite(hh) and close_price > hh:
        return 1
    if math.isfinite(ll) and close_price < ll:
        return -1
    return current_pos


def calc_position_size(
    equity: float,
    entry_price: float,
    atr_value: float,
    stop_atr: float,
    vol_mult: float,
    risk_per_trade: float,
    margin_rate: float,
    max_margin_usage: float,
    max_contracts: int,
) -> int:
    if not math.isfinite(atr_value) or atr_value <= 0:
        return 0
    stop_distance = atr_value * stop_atr
    if stop_distance <= 0:
        return 0

    risk_budget = equity * risk_per_trade
    qty_by_risk = math.floor(risk_budget / (stop_distance * vol_mult))

    margin_per_contract = entry_price * vol_mult * margin_rate
    if margin_per_contract <= 0:
        return 0
    qty_by_margin = math.floor((equity * max_margin_usage) / margin_per_contract)

    return max(0, min(qty_by_risk, qty_by_margin, max_contracts))


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    price_tick: float,
    vol_mult: float,
    initial_capital: float,
    commission_per_contract: float,
    slippage_ticks: float,
    risk_per_trade: float,
    stop_atr: float,
    rr: float,
    margin_rate: float,
    max_margin_usage: float,
    max_contracts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = initial_capital
    pos: Optional[Position] = None
    trades = []
    equity_rows = []
    slippage = slippage_ticks * price_tick

    for i in range(2, len(df)):
        dt = df.iloc[i]["dt"]
        bar_open = safe_float(df.iloc[i]["open"])
        bar_high = safe_float(df.iloc[i]["high"])
        bar_low = safe_float(df.iloc[i]["low"])
        bar_close = safe_float(df.iloc[i]["close"])
        atr_prev = safe_float(df.iloc[i - 1]["atr"])
        signal = desired_signal(df, i - 1, strategy, pos.direction if pos else 0)

        if pos and signal != pos.direction:
            exit_px = bar_open - slippage if pos.direction == 1 else bar_open + slippage
            pnl = (exit_px - pos.entry_price) * pos.qty * vol_mult * pos.direction
            cash += pnl - commission_per_contract * pos.qty
            trades.append(
                {
                    "entry_dt": df.iloc[pos.entry_idx]["dt"],
                    "exit_dt": dt,
                    "direction": "LONG" if pos.direction == 1 else "SHORT",
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_px,
                    "pnl": pnl - commission_per_contract * pos.qty,
                    "hold_bars": i - pos.entry_idx,
                    "exit_reason": "signal_flip",
                }
            )
            pos = None

        if pos is None and signal != 0:
            entry_px = bar_open + slippage if signal == 1 else bar_open - slippage
            qty = calc_position_size(
                equity=cash,
                entry_price=entry_px,
                atr_value=atr_prev,
                stop_atr=stop_atr,
                vol_mult=vol_mult,
                risk_per_trade=risk_per_trade,
                margin_rate=margin_rate,
                max_margin_usage=max_margin_usage,
                max_contracts=max_contracts,
            )
            if qty > 0:
                stop_distance = atr_prev * stop_atr
                take_distance = stop_distance * rr
                stop_px = entry_px - stop_distance if signal == 1 else entry_px + stop_distance
                take_px = entry_px + take_distance if signal == 1 else entry_px - take_distance
                cash -= commission_per_contract * qty
                pos = Position(
                    direction=signal,
                    qty=qty,
                    entry_price=entry_px,
                    entry_idx=i,
                    stop_price=stop_px,
                    take_price=take_px,
                )

        if pos is not None:
            exit_reason = None
            exit_px = None
            if pos.direction == 1:
                if bar_low <= pos.stop_price:
                    exit_reason = "stop_loss"
                    exit_px = pos.stop_price - slippage
                elif bar_high >= pos.take_price:
                    exit_reason = "take_profit"
                    exit_px = pos.take_price - slippage
            else:
                if bar_high >= pos.stop_price:
                    exit_reason = "stop_loss"
                    exit_px = pos.stop_price + slippage
                elif bar_low <= pos.take_price:
                    exit_reason = "take_profit"
                    exit_px = pos.take_price + slippage

            if exit_reason and exit_px is not None:
                pnl = (exit_px - pos.entry_price) * pos.qty * vol_mult * pos.direction
                cash += pnl - commission_per_contract * pos.qty
                trades.append(
                    {
                        "entry_dt": df.iloc[pos.entry_idx]["dt"],
                        "exit_dt": dt,
                        "direction": "LONG" if pos.direction == 1 else "SHORT",
                        "qty": pos.qty,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_px,
                        "pnl": pnl - commission_per_contract * pos.qty,
                        "hold_bars": i - pos.entry_idx + 1,
                        "exit_reason": exit_reason,
                    }
                )
                pos = None

        unrealized = 0.0
        if pos is not None:
            unrealized = (bar_close - pos.entry_price) * pos.qty * vol_mult * pos.direction
        equity_rows.append({"dt": dt, "equity": cash + unrealized})

    if pos is not None and len(df) > 0:
        dt = df.iloc[-1]["dt"]
        last_close = safe_float(df.iloc[-1]["close"])
        pnl = (last_close - pos.entry_price) * pos.qty * vol_mult * pos.direction
        cash += pnl - commission_per_contract * pos.qty
        trades.append(
            {
                "entry_dt": df.iloc[pos.entry_idx]["dt"],
                "exit_dt": dt,
                "direction": "LONG" if pos.direction == 1 else "SHORT",
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "exit_price": last_close,
                "pnl": pnl - commission_per_contract * pos.qty,
                "hold_bars": len(df) - pos.entry_idx,
                "exit_reason": "force_close_end",
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)


def calc_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    duration_seconds: int,
) -> dict:
    if equity_df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
        }

    eq = equity_df["equity"].astype(float)
    final_equity = float(eq.iloc[-1])
    total_return = final_equity / initial_capital - 1
    bar_ret = eq.pct_change().dropna()

    periods_per_year = (365 * 24 * 3600) / duration_seconds
    annual_return = (final_equity / initial_capital) ** (periods_per_year / len(eq)) - 1 if len(eq) > 1 else 0.0
    sharpe = 0.0
    if len(bar_ret) > 2 and bar_ret.std() > 0:
        sharpe = float(np.sqrt(periods_per_year) * bar_ret.mean() / bar_ret.std())

    running_max = eq.cummax()
    drawdown = eq / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    n_trades = len(trades_df)
    if n_trades == 0:
        win_rate = payoff_ratio = profit_factor = 0.0
    else:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] < 0, "pnl"]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else (float("inf") if avg_win > 0 else 0.0)
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else (float("inf") if len(wins) > 0 else 0.0)

    return {
        "trades": int(n_trades),
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
    }


def metrics_table_rows(metrics: dict) -> list[str]:
    return [
        "| Metric | Value |",
        "|---|---:|",
        f"| Trades | {metrics['trades']} |",
        f"| Win Rate | {metrics['win_rate']:.2%} |",
        f"| Payoff Ratio | {metrics['payoff_ratio']:.3f} |",
        f"| Profit Factor | {metrics['profit_factor']:.3f} |",
        f"| Max Drawdown | {metrics['max_drawdown']:.2%} |",
        f"| Total Return | {metrics['total_return']:.2%} |",
        f"| Annual Return (est.) | {metrics['annual_return']:.2%} |",
        f"| Sharpe (est.) | {metrics['sharpe']:.3f} |",
    ]


def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame, out_png: Path, title: str) -> None:
    if equity_df.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=140)
    axes[0].plot(df["dt"], df["close"], color="#1f77b4", linewidth=1.2)
    axes[0].set_title(f"{title} - Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.25)

    axes[1].plot(equity_df["dt"], equity_df["equity"], color="#2ca02c", linewidth=1.2)
    axes[1].set_title("Equity Curve")
    axes[1].set_ylabel("Equity")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def parse_list(value: str, cast=float) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def score_row(row: pd.Series) -> float:
    return (
        row["sharpe"] * 0.6
        + row["total_return"] * 6.0
        + row["win_rate"] * 0.3
        + row["profit_factor"] * 0.05
        + row["max_drawdown"] * 2.0
    )


def run_single_case(
    bars_df: pd.DataFrame,
    strategy: str,
    params: dict,
    price_tick: float,
    vol_mult: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fast_ma = int(round(float(params.get("fast_ma", args.fast_ma))))
    slow_ma = int(round(float(params.get("slow_ma", args.slow_ma))))
    breakout_window = int(round(float(params.get("breakout_window", args.breakout_window))))
    atr_window = int(round(float(params.get("atr_window", args.atr_window))))
    risk_per_trade = float(params.get("risk_per_trade", args.risk_per_trade))
    stop_atr = float(params.get("stop_atr", args.stop_atr))
    rr = float(params.get("rr", args.rr))

    with_ind = add_indicators(
        df=bars_df,
        strategy=strategy,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        breakout_window=breakout_window,
        atr_window=atr_window,
    )
    trades_df, equity_df = run_backtest(
        df=with_ind,
        strategy=strategy,
        price_tick=price_tick,
        vol_mult=vol_mult,
        initial_capital=args.initial_capital,
        commission_per_contract=args.commission,
        slippage_ticks=args.slippage_ticks,
        risk_per_trade=risk_per_trade,
        stop_atr=stop_atr,
        rr=rr,
        margin_rate=args.margin_rate,
        max_margin_usage=args.max_margin_usage,
        max_contracts=args.max_contracts,
    )
    metrics = calc_metrics(
        trades_df=trades_df,
        equity_df=equity_df,
        initial_capital=args.initial_capital,
        duration_seconds=args.duration,
    )
    return with_ind, trades_df, equity_df, metrics


def make_grid(args: argparse.Namespace) -> list[dict]:
    stop_list = parse_list(args.grid_stop_atr, float)
    rr_list = parse_list(args.grid_rr, float)
    risk_list = parse_list(args.grid_risk, float)

    grid = []
    if args.strategy == "ma":
        fast_list = parse_list(args.grid_fast_ma, int)
        slow_list = parse_list(args.grid_slow_ma, int)
        for fast, slow, stop_atr, rr, risk in itertools.product(fast_list, slow_list, stop_list, rr_list, risk_list):
            if fast >= slow:
                continue
            grid.append(
                {
                    "fast_ma": fast,
                    "slow_ma": slow,
                    "stop_atr": stop_atr,
                    "rr": rr,
                    "risk_per_trade": risk,
                    "atr_window": args.atr_window,
                    "breakout_window": args.breakout_window,
                }
            )
    else:
        breakout_list = parse_list(args.grid_breakout_window, int)
        for breakout_window, stop_atr, rr, risk in itertools.product(breakout_list, stop_list, rr_list, risk_list):
            grid.append(
                {
                    "breakout_window": breakout_window,
                    "stop_atr": stop_atr,
                    "rr": rr,
                    "risk_per_trade": risk,
                    "atr_window": args.atr_window,
                    "fast_ma": args.fast_ma,
                    "slow_ma": args.slow_ma,
                }
            )
    return grid


def optimize_parameters(
    bars_df: pd.DataFrame,
    strategy: str,
    price_tick: float,
    vol_mult: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    grid = make_grid(args)
    if not grid:
        raise RuntimeError("Grid is empty. Check grid parameter settings.")

    rows = []
    best_case = None
    for idx, params in enumerate(grid, start=1):
        _, trades_df, equity_df, metrics = run_single_case(
            bars_df=bars_df,
            strategy=strategy,
            params=params,
            price_tick=price_tick,
            vol_mult=vol_mult,
            args=args,
        )
        row = {**params, **metrics}
        row["case_id"] = idx
        row["score"] = score_row(pd.Series(row))
        rows.append(row)

        if best_case is None:
            best_case = (params, trades_df, equity_df, metrics, row["score"])
        else:
            _, _, _, _, best_score = best_case
            if row["score"] > best_score:
                best_case = (params, trades_df, equity_df, metrics, row["score"])

    compare_df = pd.DataFrame(rows)
    compare_df = compare_df.sort_values("score", ascending=False).reset_index(drop=True)

    filtered = compare_df[
        (compare_df["trades"] >= args.min_trades_filter) & (compare_df["max_drawdown"] >= -abs(args.max_dd_filter))
    ].copy()
    if not filtered.empty:
        filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
        best_case_id = int(filtered.iloc[0]["case_id"])
    else:
        best_case_id = int(compare_df.iloc[0]["case_id"])

    chosen = compare_df.loc[compare_df["case_id"] == best_case_id].iloc[0].to_dict()
    best_params = {}
    for k in ["fast_ma", "slow_ma", "breakout_window", "atr_window", "stop_atr", "rr", "risk_per_trade"]:
        if k not in chosen or pd.isna(chosen[k]):
            continue
        if k in {"fast_ma", "slow_ma", "breakout_window", "atr_window"}:
            best_params[k] = int(round(float(chosen[k])))
        else:
            best_params[k] = float(chosen[k])
    with_ind, trades_df, equity_df, metrics = run_single_case(
        bars_df=bars_df,
        strategy=strategy,
        params=best_params,
        price_tick=price_tick,
        vol_mult=vol_mult,
        args=args,
    )
    return compare_df, best_params, with_ind, trades_df, equity_df, metrics


def write_backtest_report(
    out_md: Path,
    symbol: str,
    strategy: str,
    metrics: dict,
    chart_file: str,
    trades_file: str,
    equity_file: str,
    params: dict,
) -> None:
    now = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    param_lines = [f"- `{k}`: `{v}`" for k, v in params.items()]
    lines = [
        f"# Backtest Report ({strategy})",
        "",
        f"- Time: `{now}`",
        f"- Symbol: `{symbol}`",
        "",
        "## Parameters",
        "",
        *param_lines,
        "",
        "## Metrics",
        "",
        *metrics_table_rows(metrics),
        "",
        "## Files",
        "",
        f"- Chart: `{chart_file}`",
        f"- Trades: `{trades_file}`",
        f"- Equity: `{equity_file}`",
        "",
        f"![equity_and_price]({chart_file})",
        "",
        "> Research only, not investment advice.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_grid_report(
    out_md: Path,
    symbol: str,
    strategy: str,
    compare_csv: str,
    best_params: dict,
    best_metrics: dict,
    top_df: pd.DataFrame,
) -> None:
    now = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"# Grid Optimization Report ({strategy})",
        "",
        f"- Time: `{now}`",
        f"- Symbol: `{symbol}`",
        f"- Comparison table: `{compare_csv}`",
        "",
        "## Best Parameters",
        "",
    ]
    for k, v in best_params.items():
        lines.append(f"- `{k}`: `{v}`")

    lines.extend(
        [
            "",
            "## Best Metrics",
            "",
            *metrics_table_rows(best_metrics),
            "",
            "## Top Parameter Sets",
            "",
            "| rank | case_id | score | trades | win_rate | payoff | max_dd | ret | sharpe | params |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        if strategy == "ma":
            param_text = (
                f"fast={int(row['fast_ma'])}, slow={int(row['slow_ma'])}, "
                f"stop_atr={row['stop_atr']}, rr={row['rr']}, risk={row['risk_per_trade']}"
            )
        else:
            param_text = (
                f"breakout={int(row['breakout_window'])}, stop_atr={row['stop_atr']}, "
                f"rr={row['rr']}, risk={row['risk_per_trade']}"
            )
        lines.append(
            "| {rank} | {case_id:.0f} | {score:.3f} | {trades:.0f} | {win:.2%} | {payoff:.3f} | {dd:.2%} | {ret:.2%} | {sharpe:.3f} | {params} |".format(
                rank=rank,
                case_id=row["case_id"],
                score=row["score"],
                trades=row["trades"],
                win=row["win_rate"],
                payoff=row["payoff_ratio"],
                dd=row["max_drawdown"],
                ret=row["total_return"],
                sharpe=row["sharpe"],
                params=param_text,
            )
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strategy backtest system with risk management and optional grid optimization")
    parser.add_argument("--strategy", choices=["ma", "breakout"], default="ma")
    parser.add_argument("--symbol", default="KQ.m@INE.sc")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--user", default=os.getenv("TQ_USER"))
    parser.add_argument("--password", default=os.getenv("TQ_PASSWORD"))

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

    parser.add_argument("--optimize", action="store_true", help="Enable parameter grid optimization")
    parser.add_argument("--grid-fast-ma", default="10,20,30")
    parser.add_argument("--grid-slow-ma", default="40,60,80")
    parser.add_argument("--grid-breakout-window", default="10,20,30")
    parser.add_argument("--grid-stop-atr", default="1.5,2.0,2.5")
    parser.add_argument("--grid-rr", default="1.5,2.0,2.5")
    parser.add_argument("--grid-risk", default="0.005,0.01,0.015")
    parser.add_argument("--min-trades-filter", type=int, default=20)
    parser.add_argument("--max-dd-filter", type=float, default=0.3)

    parser.add_argument("--output-dir", default="strategy_system_tool/output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.user or not args.password:
        raise SystemExit("Please provide TQ account by --user/--password or env TQ_USER/TQ_PASSWORD")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_df, price_tick, vol_mult = fetch_bars(
        user=args.user,
        password=args.password,
        symbol=args.symbol,
        duration_seconds=args.duration,
        data_length=args.bars,
    )

    stem = f"{args.strategy}_{args.symbol.replace('@', '_').replace('.', '_')}"

    if args.optimize:
        compare_df, best_params, with_ind, trades_df, equity_df, metrics = optimize_parameters(
            bars_df=bars_df,
            strategy=args.strategy,
            price_tick=price_tick,
            vol_mult=vol_mult,
            args=args,
        )

        compare_csv = out_dir / f"{stem}_grid_compare.csv"
        compare_md = out_dir / f"{stem}_grid_report.md"
        best_trades = out_dir / f"{stem}_best_trades.csv"
        best_equity = out_dir / f"{stem}_best_equity.csv"
        best_chart = out_dir / f"{stem}_best_chart.png"
        best_report = out_dir / f"{stem}_best_report.md"

        compare_df.to_csv(compare_csv, index=False, encoding="utf-8-sig")
        trades_df.to_csv(best_trades, index=False, encoding="utf-8-sig")
        equity_df.to_csv(best_equity, index=False, encoding="utf-8-sig")
        plot_results(with_ind, equity_df, best_chart, title=f"{args.symbol} | {args.strategy} | best")
        write_backtest_report(
            out_md=best_report,
            symbol=args.symbol,
            strategy=args.strategy,
            metrics=metrics,
            chart_file=best_chart.name,
            trades_file=best_trades.name,
            equity_file=best_equity.name,
            params=best_params,
        )

        top_n = min(20, len(compare_df))
        write_grid_report(
            out_md=compare_md,
            symbol=args.symbol,
            strategy=args.strategy,
            compare_csv=compare_csv.name,
            best_params=best_params,
            best_metrics=metrics,
            top_df=compare_df.head(top_n),
        )

        print(f"Grid compare table: {compare_csv}")
        print(f"Grid report: {compare_md}")
        print(f"Best params: {best_params}")
        print(
            f"Best metrics -> win_rate: {metrics['win_rate']:.2%}, payoff: {metrics['payoff_ratio']:.3f}, "
            f"max_dd: {metrics['max_drawdown']:.2%}, return: {metrics['total_return']:.2%}, sharpe: {metrics['sharpe']:.3f}"
        )
        print(f"Best artifacts: {best_trades}, {best_equity}, {best_chart}, {best_report}")
        return

    params = {
        "fast_ma": args.fast_ma,
        "slow_ma": args.slow_ma,
        "breakout_window": args.breakout_window,
        "atr_window": args.atr_window,
        "stop_atr": args.stop_atr,
        "rr": args.rr,
        "risk_per_trade": args.risk_per_trade,
    }
    with_ind, trades_df, equity_df, metrics = run_single_case(
        bars_df=bars_df,
        strategy=args.strategy,
        params=params,
        price_tick=price_tick,
        vol_mult=vol_mult,
        args=args,
    )
    trades_csv = out_dir / f"{stem}_trades.csv"
    equity_csv = out_dir / f"{stem}_equity.csv"
    chart_png = out_dir / f"{stem}_chart.png"
    report_md = out_dir / f"{stem}_report.md"

    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    equity_df.to_csv(equity_csv, index=False, encoding="utf-8-sig")
    plot_results(with_ind, equity_df, chart_png, title=f"{args.symbol} | {args.strategy}")
    write_backtest_report(
        out_md=report_md,
        symbol=args.symbol,
        strategy=args.strategy,
        metrics=metrics,
        chart_file=chart_png.name,
        trades_file=trades_csv.name,
        equity_file=equity_csv.name,
        params=params,
    )

    print(f"Trades: {trades_csv}")
    print(f"Equity: {equity_csv}")
    print(f"Chart: {chart_png}")
    print(f"Report: {report_md}")
    print(
        f"Metrics -> win_rate: {metrics['win_rate']:.2%}, payoff: {metrics['payoff_ratio']:.3f}, "
        f"max_dd: {metrics['max_drawdown']:.2%}, return: {metrics['total_return']:.2%}, sharpe: {metrics['sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()
