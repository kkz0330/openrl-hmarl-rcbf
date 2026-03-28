import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqsdk import TqApi, TqAuth


@dataclass
class StructureResult:
    regime: str
    spread: float
    spread_pct: float
    slope: float
    near_contract: str
    far_contract: str
    near_close: float
    far_close: float


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _wait_updates(api: TqApi, seconds: int = 20) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        api.wait_update(deadline=time.time() + 1)


def _extract_last_close(kline: pd.DataFrame) -> float:
    if kline is None or kline.empty:
        return math.nan
    close_series = pd.to_numeric(kline["close"], errors="coerce").dropna()
    if close_series.empty:
        return math.nan
    return float(close_series.iloc[-1])


def fetch_term_structure(
    user: str,
    password: str,
    exchange: str,
    product: str,
    max_contracts: int = 12,
) -> pd.DataFrame:
    api = TqApi(auth=TqAuth(user, password))
    try:
        symbols = list(
            api.query_quotes(
                ins_class="FUTURE",
                exchange_id=exchange,
                product_id=product,
                expired=False,
            )
        )
        if not symbols:
            raise RuntimeError(f"未查询到 {exchange}.{product} 的未到期期货合约。")

        quotes = {s: api.get_quote(s) for s in symbols}
        dailies = {s: api.get_kline_serial(s, 24 * 60 * 60, data_length=8) for s in symbols}
        _wait_updates(api, seconds=25)

        rows = []
        for symbol in symbols:
            q = quotes[symbol]
            close_price = _extract_last_close(dailies[symbol])
            if math.isnan(close_price):
                close_price = _safe_float(q.last_price)

            expire_ts = _safe_float(q.expire_datetime)
            if math.isnan(close_price) or math.isnan(expire_ts):
                continue

            delivery_year = int(q.delivery_year)
            delivery_month = int(q.delivery_month)
            rows.append(
                {
                    "symbol": symbol,
                    "instrument_id": q.instrument_id,
                    "delivery": f"{delivery_year:04d}-{delivery_month:02d}",
                    "expire_dt": pd.to_datetime(expire_ts, unit="s", utc=True).tz_convert("Asia/Shanghai"),
                    "close": close_price,
                    "open_interest": _safe_float(q.open_interest),
                    "volume_multiple": _safe_float(q.volume_multiple),
                }
            )

        df = pd.DataFrame(rows).sort_values("expire_dt").reset_index(drop=True)
        if df.empty:
            raise RuntimeError("获取行情成功但未得到有效收盘价数据，请稍后重试。")
        if max_contracts > 0:
            df = df.head(max_contracts).copy()
        return df
    finally:
        api.close()


def judge_structure(df: pd.DataFrame, threshold_pct: float = 0.01) -> StructureResult:
    near = df.iloc[0]
    far = df.iloc[-1]
    near_close = float(near["close"])
    far_close = float(far["close"])
    spread = far_close - near_close
    spread_pct = spread / near_close if near_close else 0.0

    x = np.arange(len(df), dtype=float)
    y = df["close"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(df) > 1 else 0.0

    if spread_pct > threshold_pct and slope > 0:
        regime = "升水结构（Contango）"
    elif spread_pct < -threshold_pct and slope < 0:
        regime = "贴水结构（Backwardation）"
    else:
        regime = "平坦/混合结构"

    return StructureResult(
        regime=regime,
        spread=spread,
        spread_pct=spread_pct,
        slope=slope,
        near_contract=str(near["symbol"]),
        far_contract=str(far["symbol"]),
        near_close=near_close,
        far_close=far_close,
    )


def build_strategy_text(result: StructureResult) -> List[str]:
    common_risks = "风险提示：需关注流动性差异、换月规则、保证金变化、突发供需事件和止损执行。"
    if "升水" in result.regime:
        return [
            "思路A（日历价差回归）：多近月、空远月，博弈升水收敛。",
            "思路B（展期收益）：若长期持有多头，尽量降低正向展期成本（避免被动高买远月）。",
            common_risks,
        ]
    if "贴水" in result.regime:
        return [
            "思路A（日历价差回归）：空近月、多远月，博弈贴水收敛。",
            "思路B（库存/现货紧张交易）：若近端持续偏强，可配合基本面做近强远弱跟随。",
            common_risks,
        ]
    return [
        "思路A（等待结构明朗）：平坦或混合结构下，先观察近远月价差是否突破历史区间。",
        "思路B（事件驱动）：围绕库存数据、政策、地缘事件做方向交易，期限结构只作辅助。",
        common_risks,
    ]


def plot_curve(df: pd.DataFrame, result: StructureResult, out_png: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(df["delivery"], df["close"], marker="o", linewidth=1.8, color="#1f77b4")
    for _, row in df.iterrows():
        ax.annotate(
            row["symbol"],
            (row["delivery"], row["close"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    ax.set_title(f"期限结构：{result.regime}")
    ax.set_xlabel("交割月份")
    ax.set_ylabel("最新可得日线收盘价")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_report(
    exchange: str,
    product: str,
    df: pd.DataFrame,
    result: StructureResult,
    strategy_lines: List[str],
    out_md: Path,
    out_png: Path,
) -> None:
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    table = df[["symbol", "delivery", "close", "open_interest"]].copy()
    table["close"] = table["close"].map(lambda x: f"{x:.2f}")
    table["open_interest"] = table["open_interest"].map(lambda x: f"{x:,.0f}")

    headers = list(table.columns)
    md_table = []
    md_table.append("| " + " | ".join(headers) + " |")
    md_table.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in table.iterrows():
        values = [str(row[h]) for h in headers]
        md_table.append("| " + " | ".join(values) + " |")

    lines = [
        f"# {exchange}.{product} 期限结构分析",
        "",
        f"- 生成时间：`{generated_at}`",
        f"- 结构判断：**{result.regime}**",
        f"- 近月 `{result.near_contract}` 收盘价：`{result.near_close:.2f}`",
        f"- 远月 `{result.far_contract}` 收盘价：`{result.far_close:.2f}`",
        f"- 远近月价差（远-近）：`{result.spread:.2f}`（`{result.spread_pct:.2%}`）",
        f"- 曲线斜率（线性拟合）：`{result.slope:.4f}`",
        "",
        "## 期限结构图",
        "",
        f"![term_structure]({out_png.name})",
        "",
        "## 合约收盘价表",
        "",
        *md_table,
        "",
        "## 可考虑的交易策略（研究用途）",
        "",
    ]
    for item in strategy_lines:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("> 以上策略仅供研究，不构成投资建议。")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取某品种多月份收盘价并分析期限结构")
    parser.add_argument("--exchange", default="INE", help="交易所代码，例如 INE/SHFE/DCE")
    parser.add_argument("--product", default="sc", help="品种代码，例如 sc/cu/m")
    parser.add_argument("--user", default=os.getenv("TQ_USER"), help="天勤账号，可用环境变量 TQ_USER")
    parser.add_argument("--password", default=os.getenv("TQ_PASSWORD"), help="天勤密码，可用环境变量 TQ_PASSWORD")
    parser.add_argument("--max-contracts", type=int, default=12, help="最多分析多少个近月合约")
    parser.add_argument(
        "--output-dir",
        default="term_structure_tool/output",
        help="输出目录，默认 term_structure_tool/output",
    )
    args = parser.parse_args()

    if not args.user or not args.password:
        raise SystemExit("请通过 --user/--password 或环境变量 TQ_USER/TQ_PASSWORD 提供天勤账号。")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_term_structure(
        user=args.user,
        password=args.password,
        exchange=args.exchange,
        product=args.product,
        max_contracts=args.max_contracts,
    )
    result = judge_structure(df)
    strategies = build_strategy_text(result)

    chart_path = output_dir / f"{args.exchange}_{args.product}_term_structure.png"
    csv_path = output_dir / f"{args.exchange}_{args.product}_term_structure.csv"
    report_path = output_dir / f"{args.exchange}_{args.product}_analysis.md"

    plot_curve(df, result, chart_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_report(
        exchange=args.exchange,
        product=args.product,
        df=df,
        result=result,
        strategy_lines=strategies,
        out_md=report_path,
        out_png=chart_path,
    )

    print(f"已输出 CSV: {csv_path}")
    print(f"已输出 图表: {chart_path}")
    print(f"已输出 报告: {report_path}")
    print(f"结构判断: {result.regime} | 价差(远-近): {result.spread:.2f} ({result.spread_pct:.2%})")


if __name__ == "__main__":
    main()
