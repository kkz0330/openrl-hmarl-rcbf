# INE 原油行情可视化与预测

本项目基于天勤量化 `tqsdk` 读取 INE 原油（`sc`）行情，生成一个可直接打开的网页报告，展示：

- 现在价格（主连）
- 开盘价格（主连）
- 各合约增减仓（以及乘以合约乘数后的桶数）
- 期限结构（按交割月）
- 行情图（1小时K线）
- 24小时价格预测（点估计 + 区间）

## 目录结构

- `ine_crude_dashboard.py`：主程序，连接天勤接口并生成网页
- `ine_crude_dashboard.html`：示例输出网页
- `generate_assignment_report.py`、`assignment_report.pdf`：原项目已有文件

## 环境要求

- Python 3.10+
- 可访问天勤行情服务网络
- 有效天勤账号密码

安装依赖：

```bash
pip install tqsdk pandas plotly flask numpy
```

## 使用方法

在项目根目录运行：

```bash
python ine_crude_dashboard.py --user <你的天勤账号> --password <你的天勤密码>
```

可选参数：

- `--output`：输出 HTML 路径（默认 `ine_crude_dashboard.html`）

示例：

```bash
python ine_crude_dashboard.py --user demo --password demo --output output.html
```

## 如何打开网页

方式1：直接双击 HTML 文件  
方式2：PowerShell 命令打开

```powershell
Start-Process "C:\Users\shirosk\Documents\New project\ine_crude_dashboard.html"
```

方式3：本地静态服务访问

```powershell
cd "C:\Users\shirosk\Documents\New project"
python -m http.server 8000
```

浏览器访问：`http://localhost:8000/ine_crude_dashboard.html`

## 计算口径说明

- 增减仓（手）=`open_interest - pre_open_interest`
- 增减仓乘数后（桶）=`增减仓(手) * volume_multiple`
- 当前最明显增仓/减仓合约：按 `增减仓(手)` 在全体未到期合约中取最大/最小
- 预测模型：默认使用最近 240 根 1 小时K线做线性回归外推 24 小时，并给出基于残差波动的区间

## 注意事项

- 当前主图使用 `KQ.m@INE.sc`（主连），换月时可能出现价格跳变，这是主连合约机制本身导致。
- 预测结果仅用于演示，不构成任何投资建议。
- 若网络或账号认证失败，程序会报错退出，请先检查网络和账号状态。

