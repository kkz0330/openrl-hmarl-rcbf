# TqSdk Codex 开始前准备（已落地）

本目录按文档 `tqsdk_codex.html#tqsdk-codex` 的“开始前准备”完成了本地初始化。

## 目录结构

- `repos/tqsdk-python`: 官方源码仓库
- `.venv`: 本地 Python 虚拟环境
- `skills/tqsdk-trading-and-data`: 下载的 skills 压缩包及解压目录
- `scripts/setup_env.ps1`: 一键重做环境脚本
- `scripts/verify_preparation.py`: 开始前准备自检脚本
- `PREP_RESULT.md`: 本次准备结果记录

## 已完成事项

1. 源码仓库准备：已克隆 `tqsdk-python`
2. Python 环境准备：已创建 `.venv` 并安装依赖
3. 本地源码安装：已执行 `pip install -e repos/tqsdk-python`
4. Skills 准备：已下载并安装到 `C:\Users\shirosk\.codex\skills\tqsdk-trading-and-data`

## 运行自检

```powershell
cd "C:\Users\shirosk\Documents\New project\tqsdk_codex_start_prep"
.\.venv\Scripts\python.exe .\scripts\verify_preparation.py
```

## 后续使用建议

1. 在 Codex 中打开目录：`C:\Users\shirosk\Documents\New project\tqsdk_codex_start_prep\repos\tqsdk-python`
2. 需要登录行情/交易时，先设置环境变量：

```powershell
$env:TQ_USER="你的天勤账号"
$env:TQ_PASSWORD="你的天勤密码"
```

