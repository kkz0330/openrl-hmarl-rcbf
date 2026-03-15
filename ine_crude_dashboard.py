import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqsdk import TqApi, TqAuth


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _wait_for_data(api: TqApi, max_seconds: int = 20) -> None:
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        api.wait_update(deadline=time.time() + 1)


def _to_shanghai(ns_epoch: pd.Series) -> pd.Series:
    values = pd.to_numeric(ns_epoch, errors="coerce").fillna(0).astype("int64")
    return pd.to_datetime(values, unit="ns", utc=True).dt.tz_convert("Asia/Shanghai")


def fetch_ine_crude_snapshot(user: str, password: str) -> dict:
    api = TqApi(auth=TqAuth(user, password))
    try:
        contracts = list(
            api.query_quotes(
                ins_class="FUTURE",
                exchange_id="INE",
                product_id="sc",
                expired=False,
            )
        )
        main_symbol = "KQ.m@INE.sc"
        main_quote = api.get_quote(main_symbol)
        contract_quotes = {symbol: api.get_quote(symbol) for symbol in contracts}
        kline_hour = api.get_kline_serial(main_symbol, 60 * 60, data_length=240)
        _wait_for_data(api, max_seconds=25)

        rows: list[dict] = []
        for symbol, quote in contract_quotes.items():
            last_price = _safe_float(quote.last_price)
            open_price = _safe_float(quote.open)
            open_interest = _safe_float(quote.open_interest)
            pre_open_interest = _safe_float(quote.pre_open_interest)
            multiplier = _safe_float(quote.volume_multiple)
            expire_ts = _safe_float(quote.expire_datetime)
            if math.isnan(last_price) or math.isnan(expire_ts):
                continue
            oi_change = open_interest - pre_open_interest
            oi_change_multiplied = oi_change * multiplier
            notional_change = oi_change_multiplied * last_price
            rows.append(
                {
                    "symbol": symbol,
                    "instrument_id": quote.instrument_id,
                    "delivery": f"{int(quote.delivery_year):04d}-{int(quote.delivery_month):02d}",
                    "expire_dt": pd.to_datetime(expire_ts, unit="s", utc=True).tz_convert("Asia/Shanghai"),
                    "last_price": last_price,
                    "open_price": open_price,
                    "open_interest": open_interest,
                    "pre_open_interest": pre_open_interest,
                    "oi_change_lots": oi_change,
                    "multiplier": multiplier,
                    "oi_change_multiplied": oi_change_multiplied,
                    "oi_notional_change": notional_change,
                }
            )

        term_df = pd.DataFrame(rows).sort_values("expire_dt").reset_index(drop=True)
        if term_df.empty:
            raise RuntimeError("未获取到 INE.sc 合约行情。")

        rising_row = term_df.loc[term_df["oi_change_lots"].idxmax()].to_dict()
        falling_row = term_df.loc[term_df["oi_change_lots"].idxmin()].to_dict()

        kl = kline_hour.copy()
        kl["dt"] = _to_shanghai(kl["datetime"])
        kl = kl.dropna(subset=["close"]).reset_index(drop=True)

        y = pd.to_numeric(kl["close"], errors="coerce").dropna().to_numpy()
        n = len(y)
        horizon = 24
        if n >= 30:
            x = np.arange(n, dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (slope * x + intercept)
            sigma = float(np.nanstd(residual))
            fx = np.arange(n - 1, n + horizon, dtype=float)
            future_prices = slope * fx + intercept
            model_name = "线性回归(240根1小时K线)"
        else:
            sigma = float(np.nanstd(y))
            base = float(np.nanmean(y))
            future_prices = np.full(horizon + 1, base)
            model_name = "均值外推(样本不足)"

        last_ts = kl["dt"].iloc[-1]
        future_times = [last_ts + pd.Timedelta(hours=i) for i in range(horizon + 1)]
        forecast_price_24h = float(future_prices[-1])
        forecast_low = forecast_price_24h - 1.96 * sigma
        forecast_high = forecast_price_24h + 1.96 * sigma

        return {
            "main_quote": {
                "symbol": main_symbol,
                "underlying_symbol": main_quote.underlying_symbol,
                "last_price": _safe_float(main_quote.last_price),
                "open_price": _safe_float(main_quote.open),
                "datetime": main_quote.datetime,
            },
            "term_df": term_df,
            "kline_df": kl,
            "rising_contract": rising_row,
            "falling_contract": falling_row,
            "forecast": {
                "model": model_name,
                "horizon_hours": horizon,
                "future_times": future_times,
                "future_prices": future_prices.tolist(),
                "point_estimate": forecast_price_24h,
                "band_low": forecast_low,
                "band_high": forecast_high,
                "sigma": sigma,
            },
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai"),
        }
    finally:
        api.close()


def build_html(snapshot: dict, output_path: Path) -> None:
    term_df = snapshot["term_df"].copy()
    kl = snapshot["kline_df"].copy()
    fc = snapshot["forecast"]
    main_quote = snapshot["main_quote"]

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Candlestick(
            x=kl["dt"],
            open=kl["open"],
            high=kl["high"],
            low=kl["low"],
            close=kl["close"],
            name="INE原油主连 1H",
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=fc["future_times"],
            y=fc["future_prices"],
            mode="lines",
            line={"color": "#D62728", "dash": "dash"},
            name=f"{fc['horizon_hours']}小时预测",
        )
    )
    fig_price.update_layout(
        title="原油主连行情与预测",
        xaxis_title="时间(Asia/Shanghai)",
        yaxis_title="价格(元/桶)",
        xaxis_rangeslider_visible=False,
        height=520,
        template="plotly_white",
    )

    fig_term = go.Figure()
    fig_term.add_trace(
        go.Scatter(
            x=term_df["delivery"],
            y=term_df["last_price"],
            mode="lines+markers+text",
            text=term_df["symbol"],
            textposition="top center",
            name="期限结构",
        )
    )
    fig_term.update_layout(
        title="INE.sc 期限结构(按交割月)",
        xaxis_title="交割月",
        yaxis_title="最新价(元/桶)",
        template="plotly_white",
        height=420,
    )

    oi_plot_df = term_df.sort_values("oi_change_lots", ascending=False).copy()
    fig_oi = go.Figure()
    fig_oi.add_trace(
        go.Bar(
            x=oi_plot_df["symbol"],
            y=oi_plot_df["oi_change_multiplied"],
            marker_color=np.where(oi_plot_df["oi_change_multiplied"] >= 0, "#2CA02C", "#D62728"),
            name="增减仓 x 合约乘数",
        )
    )
    fig_oi.update_layout(
        title="合约增减仓(手)乘以合约乘数后的变化量",
        xaxis_title="合约",
        yaxis_title="变化量(桶)",
        template="plotly_white",
        height=420,
    )

    table_df = term_df[
        [
            "symbol",
            "delivery",
            "last_price",
            "open_price",
            "open_interest",
            "pre_open_interest",
            "oi_change_lots",
            "multiplier",
            "oi_change_multiplied",
        ]
    ].copy()
    table_df.columns = [
        "合约",
        "交割月",
        "最新价",
        "开盘价",
        "当前持仓",
        "昨持仓",
        "增减仓(手)",
        "合约乘数",
        "增减仓×乘数(桶)",
    ]
    for col in ["最新价", "开盘价"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2f}")
    for col in ["当前持仓", "昨持仓", "增减仓(手)", "合约乘数", "增减仓×乘数(桶)"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:,.0f}")
    table_html = table_df.to_html(index=False, classes="table", justify="center", border=0)

    rising = snapshot["rising_contract"]
    falling = snapshot["falling_contract"]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>INE 原油走势与预测</title>
  <style>
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      background: linear-gradient(135deg, #f5f7fa, #e3ecf7);
      color: #111827;
      padding: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
      padding: 14px;
    }}
    .label {{
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .small {{
      font-size: 13px;
      color: #374151;
      margin-top: 6px;
      line-height: 1.5;
    }}
    .chart {{
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
      margin-bottom: 16px;
      padding: 8px;
    }}
    .table-wrap {{
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
      padding: 12px;
      overflow-x: auto;
    }}
    table.table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    table.table th, table.table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px;
      text-align: center;
      white-space: nowrap;
    }}
    table.table th {{
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card">
      <div class="label">现在价格（主连）</div>
      <div class="value">{main_quote["last_price"]:.2f}</div>
      <div class="small">标的合约: {main_quote["underlying_symbol"]}</div>
    </div>
    <div class="card">
      <div class="label">主连开盘价</div>
      <div class="value">{main_quote["open_price"]:.2f}</div>
      <div class="small">行情时间: {main_quote["datetime"]}</div>
    </div>
    <div class="card">
      <div class="label">24小时预测价</div>
      <div class="value">{fc["point_estimate"]:.2f}</div>
      <div class="small">区间: [{fc["band_low"]:.2f}, {fc["band_high"]:.2f}]<br/>模型: {fc["model"]}</div>
    </div>
    <div class="card">
      <div class="label">增仓最明显合约</div>
      <div class="value">{rising["symbol"]}</div>
      <div class="small">增减仓(手): {rising["oi_change_lots"]:,.0f}<br/>乘数后(桶): {rising["oi_change_multiplied"]:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">减仓最明显合约</div>
      <div class="value">{falling["symbol"]}</div>
      <div class="small">增减仓(手): {falling["oi_change_lots"]:,.0f}<br/>乘数后(桶): {falling["oi_change_multiplied"]:,.0f}</div>
    </div>
  </div>

  <div class="chart">{fig_price.to_html(full_html=False, include_plotlyjs="cdn")}</div>
  <div class="chart">{fig_term.to_html(full_html=False, include_plotlyjs=False)}</div>
  <div class="chart">{fig_oi.to_html(full_html=False, include_plotlyjs=False)}</div>

  <div class="table-wrap">
    <h3>各合约快照（含增减仓 x 合约乘数）</h3>
    {table_html}
    <p style="font-size:12px;color:#6b7280;">生成时间: {snapshot["generated_at"].strftime("%Y-%m-%d %H:%M:%S %Z")}</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用天勤SDK生成 INE 原油行情网页报告")
    parser.add_argument("--user", required=True, help="天勤账号")
    parser.add_argument("--password", required=True, help="天勤密码")
    parser.add_argument(
        "--output",
        default="ine_crude_dashboard.html",
        help="输出 HTML 文件路径，默认: ine_crude_dashboard.html",
    )
    args = parser.parse_args()

    snapshot = fetch_ine_crude_snapshot(args.user, args.password)
    output_path = Path(args.output).resolve()
    build_html(snapshot, output_path)
    print(f"已生成网页报告: {output_path}")


if __name__ == "__main__":
    main()
