import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _linear_trend_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) < 2:
        return np.full(horizon, y[-1] if len(y) else math.nan)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fx = np.arange(len(y), len(y) + horizon, dtype=float)
    return slope * fx + intercept


def _ewma_level_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) == 0:
        return np.full(horizon, math.nan)
    span = max(3, min(24, len(y)))
    level = float(pd.Series(y).ewm(span=span, adjust=False).mean().iloc[-1])
    return np.full(horizon, level)


def _ar1_log_return_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) < 4 or np.any(y <= 0):
        return np.full(horizon, y[-1] if len(y) else math.nan)

    r = np.diff(np.log(y))
    if len(r) < 2:
        return np.full(horizon, y[-1])

    x = r[:-1]
    target = r[1:]
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    c, phi = float(coef[0]), float(coef[1])

    last_r = float(r[-1])
    future_r = []
    for _ in range(horizon):
        next_r = c + phi * last_r
        future_r.append(next_r)
        last_r = next_r

    log_path = np.log(y[-1]) + np.cumsum(np.array(future_r, dtype=float))
    return np.exp(log_path)


def _holt_linear_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) < 2:
        return np.full(horizon, y[-1] if len(y) else math.nan)

    alphas = [0.2, 0.4, 0.6, 0.8]
    betas = [0.2, 0.4, 0.6, 0.8]
    best_rmse = float("inf")
    best_level = float(y[0])
    best_trend = float(y[1] - y[0])

    for alpha in alphas:
        for beta in betas:
            level = float(y[0])
            trend = float(y[1] - y[0])
            errors = []
            for i in range(1, len(y)):
                fitted = level + trend
                errors.append(y[i] - fitted)
                prev_level = level
                level = alpha * y[i] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend

            candidate_rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
            if candidate_rmse < best_rmse:
                best_rmse = candidate_rmse
                best_level = level
                best_trend = trend

    preds = np.array([best_level + (i + 1) * best_trend for i in range(horizon)], dtype=float)
    return np.maximum(preds, 1e-6)


def _evaluate_models(y: np.ndarray, horizon: int) -> dict:
    models: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "LinearTrend": _linear_trend_forecast,
        "EWMA": _ewma_level_forecast,
        "AR1LogReturn": _ar1_log_return_forecast,
        "HoltLinear": _holt_linear_forecast,
    }

    n = len(y)
    if n < 20:
        raise RuntimeError("K线样本过少，无法进行多模型预测。")

    val_size = min(24, max(6, n // 5))
    train = y[:-val_size]
    val = y[-val_size:]

    results: dict[str, dict] = {}
    for name, func in models.items():
        if len(train) < 6:
            val_pred = np.full(len(val), train[-1] if len(train) else math.nan)
        else:
            val_pred = func(train, len(val))
        val_pred = np.asarray(val_pred, dtype=float)

        mae = _mae(val, val_pred)
        rmse = _rmse(val, val_pred)
        error_std = float(np.std(val - val_pred))

        full_pred = np.asarray(func(y, horizon), dtype=float)
        point = float(full_pred[-1]) if len(full_pred) else math.nan
        band_low = point - 1.96 * error_std
        band_high = point + 1.96 * error_std

        results[name] = {
            "val_mae": mae,
            "val_rmse": rmse,
            "error_std": error_std,
            "forecast": full_pred,
            "point_estimate": point,
            "band_low": band_low,
            "band_high": band_high,
        }

    best_model = min(results.keys(), key=lambda m: results[m]["val_mae"])
    return {"models": results, "best_model": best_model, "horizon": horizon}


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
                    "oi_change_multiplied": oi_change * multiplier,
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
        close_prices = pd.to_numeric(kl["close"], errors="coerce").dropna().to_numpy(dtype=float)

        forecast = _evaluate_models(close_prices, horizon=24)
        last_ts = kl["dt"].iloc[-1]
        future_times = [last_ts + pd.Timedelta(hours=i) for i in range(1, forecast["horizon"] + 1)]

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
            "forecast": forecast,
            "future_times": future_times,
            "generated_at": pd.Timestamp.now(tz="Asia/Shanghai"),
        }
    finally:
        api.close()


def build_html(snapshot: dict, output_path: Path) -> None:
    term_df = snapshot["term_df"].copy()
    kl = snapshot["kline_df"].copy()
    main_quote = snapshot["main_quote"]
    forecast = snapshot["forecast"]
    future_times = snapshot["future_times"]
    best_model = forecast["best_model"]
    best_info = forecast["models"][best_model]

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
    model_colors = {
        "LinearTrend": "#D62728",
        "EWMA": "#9467BD",
        "AR1LogReturn": "#1F77B4",
        "HoltLinear": "#2CA02C",
    }
    last_price = float(pd.to_numeric(kl["close"], errors="coerce").iloc[-1])
    last_time = kl["dt"].iloc[-1]
    for name, info in forecast["models"].items():
        line_x = [last_time] + future_times
        line_y = [last_price] + info["forecast"].tolist()
        fig_price.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": model_colors.get(name, "#333333"), "dash": "dash"},
                name=f"{name} 24h预测",
            )
        )
    fig_price.update_layout(
        title=f"原油主连行情与多模型预测（最佳: {best_model}）",
        xaxis_title="时间(Asia/Shanghai)",
        yaxis_title="价格(元/桶)",
        xaxis_rangeslider_visible=False,
        height=560,
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
        title="INE.sc 期限结构（按交割月）",
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
        title="合约增减仓（手）乘以合约乘数后的变化量",
        xaxis_title="合约",
        yaxis_title="变化量(桶)",
        template="plotly_white",
        height=420,
    )

    term_table = term_df[
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
    term_table.columns = [
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
        term_table[col] = term_table[col].map(lambda x: f"{x:.2f}")
    for col in ["当前持仓", "昨持仓", "增减仓(手)", "合约乘数", "增减仓×乘数(桶)"]:
        term_table[col] = term_table[col].map(lambda x: f"{x:,.0f}")
    term_table_html = term_table.to_html(index=False, classes="table", justify="center", border=0)

    metrics_rows = []
    for model_name, info in forecast["models"].items():
        metrics_rows.append(
            {
                "模型": model_name,
                "验证MAE": f"{info['val_mae']:.3f}",
                "验证RMSE": f"{info['val_rmse']:.3f}",
                "24h预测价": f"{info['point_estimate']:.2f}",
                "95%区间": f"[{info['band_low']:.2f}, {info['band_high']:.2f}]",
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("验证MAE")
    metrics_html = metrics_df.to_html(index=False, classes="table", justify="center", border=0)

    rising = snapshot["rising_contract"]
    falling = snapshot["falling_contract"]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>INE 原油走势与多模型预测</title>
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
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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
      margin-bottom: 16px;
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
      <div class="label">最佳模型</div>
      <div class="value">{best_model}</div>
      <div class="small">按验证集 MAE 选择</div>
    </div>
    <div class="card">
      <div class="label">最佳模型24h预测</div>
      <div class="value">{best_info["point_estimate"]:.2f}</div>
      <div class="small">区间: [{best_info["band_low"]:.2f}, {best_info["band_high"]:.2f}]</div>
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
    <h3>多模型预测结果（按验证MAE排序）</h3>
    {metrics_html}
  </div>

  <div class="table-wrap">
    <h3>各合约快照（含增减仓 x 合约乘数）</h3>
    {term_table_html}
    <p style="font-size:12px;color:#6b7280;">生成时间: {snapshot["generated_at"].strftime("%Y-%m-%d %H:%M:%S %Z")}</p>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用天勤SDK生成 INE 原油行情网页（含多模型预测）")
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
