# GCBF+ 分散式 CBF 基线（当前项目）

该基线是一个独立模块，不依赖分层 RL 训练，直接在当前 `MultiUAV2DEnv` 中运行固定参数的分散式 CBF-QP 控制器。

## 代码位置

- 控制器：`hmarl_cbf/baselines/distributed_cbf.py`
- 运行脚本：`hmarl_cbf/train/run_distributed_cbf_baseline.py`
- 默认配置：`configs/hmarl_cbf/baseline_distributed_cbf.yaml`

## CBF 模式

配置项 `baseline.cbf_mode` 支持：

- `distributed_hocbf54`（默认，GCBF+ 54 式风格 reciprocal CBF）
- `distributed_gcbfplus`
- `distributed_ecbf`

## 运行命令（WSL）

```bash
cd "/mnt/c/Users/shirosk/Documents/New project"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gcbfplus

python -m hmarl_cbf.train.run_distributed_cbf_baseline \
  --config configs/hmarl_cbf/baseline_distributed_cbf.yaml \
  --run-name gcbfplus_distributed_baseline \
  --episodes 20 \
  --render-gif \
  --fps 8
```

## 输出

- `artifacts/hmarl_cbf_baseline/<run-name>/episode_results.csv`
- `artifacts/hmarl_cbf_baseline/<run-name>/summary.json`
- `artifacts/hmarl_cbf_baseline/<run-name>/media/episode_XXX.gif`

