# HMARL-CBF 算法实现全说明（当前代码版）

本文档对应当前仓库 `hmarl_cbf/` 的实际实现，用于统一说明：
- 代码结构
- 算法逻辑结构（上层/下层/环境/技能/同步协议）
- 关键数学式子（动力学、奖励、MAPPO、QP/CBF/CLF、下层损失）
- 默认配置与可切换项
- 训练与评测入口

## 1. 代码结构总览

主模块：`hmarl_cbf/`

- `env/`
  - `multi_uav_2d_env.py`: 2D 多智能体环境（双积分器、障碍、奖励、终止）
  - `lidar.py`: LiDAR 扫描模型
  - `observation.py`: 高层/低层观测拼装
- `skills/`
  - `library.py`: 技能定义（turn_left/right, accelerate, decelerate, cruise, hover）
  - `runtime.py`: 每个智能体的技能运行时（激活、计时、终止、内在奖励）
  - `termination.py`: 技能终止统一入口
- `policies/`
  - `high_level.py`: 上层离散技能策略 + value 网络
  - `low_level_qp.py`: 下层 QP 参数网络 + 低层 value 头
- `control/`
  - `constraint_builder.py`: 构建硬 CBF + 软 CLF + 限幅约束
  - `qp_solver.py`: 非可微 QP 求解（cvxpylayers 推理调用 + stub fallback）
  - `diff_qp.py`: 可微 QP（KKT 反传路径）
  - `low_level_controller.py`: 下层控制链路（技能参考融合 + QP）
  - `sync_coordinator.py`: sync/async 技能切换协调器
- `high_level/`
  - `mappo.py`: 上层 MAPPO 更新器
- `buffer/`
  - `hier_rollout_buffer.py`: 分层回放（步级低层 + 轮次级高层）
- `train/`
  - `trainer_sync_onpolicy.py`: 主训练器（rollout/update/eval）
  - `run_sync_onpolicy.py`: 训练入口
  - `eval_checkpoint_random.py`: 随机场景评测入口
- `types.py`
  - 数据结构定义（状态、观测、QP、transition 等）

配置目录：`configs/hmarl_cbf/`

- `default_async_onpolicy_gcbfplus.yaml`（当前常用）
- `default_sync_onpolicy.yaml`
- `default_sync_onpolicy_gcbfplus.yaml`

## 2. 任务与系统建模

### 2.1 状态与控制

每个智能体状态（2D 双积分器）：

- `x_i = [p_x, p_y, v_x, v_y]`
- 控制 `u_i = [a_x, a_y]`

环境离散更新（见 `env/multi_uav_2d_env.py`）：

- `v_{t+1} = clip(v_t + u_t * dt, -velocity_limit, velocity_limit)`
- `p_{t+1} = p_t + v_{t+1} * dt`
- 并对动作先做 `action_limit` 限幅。

### 2.2 终止条件

单智能体到达条件：

- `||p - p_goal|| <= goal_threshold`
- 且 `||v|| <= goal_speed_threshold`

episode 终止：

- 全体到达，或
- `terminate_on_collision=true` 且任意不安全（碰撞或出界），或
- 到达 `horizon`（truncated）。

## 3. 观测结构（部分可观测）

每个智能体高层观测 `obs_high`：

- `self_state = [p_x, p_y, v_x, v_y]`
- `goal_relative = goal - position`
- `neighbor_summary`（最多 `max_neighbors`，每个邻居 4 维：相对位置2 + 相对速度2）

每个智能体低层观测 `obs_low`：

- `self_state`
- `goal_relative`
- `lidar_scan`（`N_beam` 束，归一化到 `[0,1]`）
- `neighbor_summary`

LiDAR 模型（`env/lidar.py`）：

- 每束做射线-圆障碍/邻居求交，取最近距离
- 未命中返回 `max_range`
- 可选高斯噪声 `N(0, sigma^2)`

## 4. 技能层（7 元组映射）

实现的 6 个技能：

- `turn_left`, `turn_right`, `accelerate`, `decelerate`, `cruise`, `hover`

每个技能在 `SkillSpec` 中包含：

- 启动集函数 `initiation_set_fn`
- 中止集函数 `termination_set_fn`
- 最大时长 `max_duration`
- 中止函数 `termination_fn(state, ctx, tau)`
- 安全集函数 `safety_constraints_fn`
- 内在奖励函数 `intrinsic_reward_fn`
- 安全技能策略（参考动作）`safe_skill_policy`

技能终止统一逻辑：

- `beta_i = termination_fn(...)`
- 实际实现一般为：`in termination_set OR tau >= max_duration`

## 5. 分层控制与执行协议

### 5.1 总体结构

每步链路：

1. 上层（按轮次）给技能 `z_i`
2. 下层网络输入 `(obs_low, z_i)` 输出 `QPParam`
3. 技能策略输出 `u_ref_skill`
4. 融合参考：
   - `u_ref_fused = w_skill * u_ref_skill + (1 - w_skill) * u_ref_policy`
5. 构建 QP 约束（CBF/CLF/输入限幅）
6. 求解 QP 得到执行动作 `u`

### 5.2 sync/async 切换（`sync_coordinator.py`）

- `mode=sync`：
  - `sync_switch = all(beta_i) OR max(tau_i) >= T_sync_max`
  - 切换时全体重置本轮计时
- `mode=async`：
  - 智能体 `i` 若 `beta_i=1` 或 `tau_i >= T_sync_max` 则单独切换技能
  - 未触发切换的智能体保持当前技能

## 6. 上层：MAPPO（离散技能）

策略网络（`policies/high_level.py`）：

- backbone: `MLP(Tanh)` -> `actor logits` + `value`
- 动作为离散技能 id（Categorical）

MAPPO 更新（`high_level/mappo.py`）：

- 比率：`r = exp(logp_new - logp_old)`
- actor：
  - `L_actor = -E[min(r*A, clip(r, 1-e, 1+e)*A)]`
- value（clipped value loss）：
  - `V_clip = V_old + clip(V_new - V_old, -e, e)`
  - `L_value = 0.5 * E[max((V_new-R)^2, (V_clip-R)^2)]`
- 熵正则：`H`
- 总损失：
  - `L_high = L_actor + c_v * L_value - c_ent * H`

高层样本来自高层 option 片段（非每步）：`(obs_high, z, logp, value, return_ext, advantage, value_target)`

## 7. 下层：参数化 QP + 可微求解

### 7.1 下层网络输出

`low_level_qp.py` 输出：

- `u_ref`（2维）
- `r_diag`（目标二次项对角，softplus 保正）
- `w_clf`（CLF 松弛权重，softplus 保正）
- `cbf_k0, cbf_k1`（CBF 增益，softplus 保正）
- `clf_k`（CLF 增益，softplus 保正）
- 低层 value 头：`low_value(obs_low, skill_id)`

### 7.2 QP 标准型

`constraint_builder.py` 构建问题：

- 目标（等价）：
  - `min 0.5 * u^T diag(r_diag) u + f^T u + w_clf * delta`
  - 当 `f = -diag(r_diag) * u_ref` 时等价于 `0.5||u-u_ref||_R^2 + w_clf*delta`
- 约束：
  - 硬 CBF：`A_cbf u <= b_cbf`
  - 软 CLF：`A_clf u <= b_clf + delta`
  - 输入约束：`u_min <= u <= u_max`（可关）
  - `delta >= 0`

### 7.3 CBF 形式（可切换）

配置键：`cbf_mode`

1. `distributed_ecbf`（默认基线）

- 对邻居/障碍定义
  - `h = ||p_rel||^2 - d_safe^2`
  - `h_dot = 2 p_rel^T v_rel`
  - `const = 2 ||v_rel||^2`
- 线性化后约束：
  - `A = -2 p_rel`
  - `b = const + k1*h_dot + k0*h`

2. `distributed_gcbfplus`（GCBF+ 风格分布式责任）

- `h0 = ||p_rel||^2 - d_safe^2`
- `h0_dot = 2 p_rel^T v_rel`
- `h1 = h0_dot + alpha0*h0`, `alpha0 = k0`
- `Lf(h1) = 2||v_rel||^2 + 2*alpha0*(p_rel^T v_rel)`
- 约束采用责任系数：
  - `A = -2 p_rel`
  - `b = share * (Lf(h1) + alpha1*h1)`, `alpha1 = k1`
- `share` 分别由 `cbf_share_agent` / `cbf_share_obs` 控制

3. `distributed_hocbf54`（论文 Eq.(54) 风格）

- Reciprocal barrier 近似：
  - `h = sqrt(4*u_max*(||p_rel||-d_safe)) + n^T v_rel`
  - `n = p_rel / ||p_rel||`
- 对应线性行 `A u <= b` 由 `_build_hocbf54_row` / `_build_hocbf54_row_torch` 实现。

### 7.4 CLF 形式

- 期望速度 `v_des`：
  - 若技能给了 `clf_v_des_vector`，则直接使用
  - 否则按目标方向构造：`v_des = s_ref * goal_dir`
- `V = 0.5 ||v - v_des||^2`
- `A_clf = (v - v_des)^T`
- `b_clf = -k_clf * V`
- 在 QP 中通过 `delta` 软化。

### 7.5 求解器策略

推理 QP（`qp_solver.py`）：

- 先 ECOS
- 失败则 SCS
- 再失败用 deterministic stub（按目标项闭式+限幅）

可微 QP（`diff_qp.py`）用于训练反传：

- 同样 ECOS -> SCS -> 张量 fallback
- 保留对 `r_diag / w_clf / cbf_k* / clf_k / u_ref` 的梯度路径

## 8. 下层训练模式（可切换）

配置键：`train.low_update_mode`

### 8.1 `target_regression`

损失（每样本）：

- `L_low = 0.5 * ||u_qp - u_target||^2`

其中：

- `u_target = u_exec + scale * advantage * goal_dir`（并可裁剪到输入边界）

### 8.2 `onpolicy_ppo`

此模式在 rollout 时构造低层策略分布：

- 先计算 QP 均值动作 `mu`（由网络参数 + QP）
- 采样代理动作：`a_sample = mu + eps, eps~N(0, sigma^2)`
- 记录 `logp_old` 与 `value_old`

更新时：

- `r = exp(logp_new - logp_old)`
- `L_actor = -min(r*A, clip(r,1-e,1+e)*A)`
- `L_value = 0.5*(V-R)^2`
- `L_low = L_actor + c_v*L_value - c_ent*H`

注意：当前低层回报默认主要是内在奖励，因为：

- `low_ext_reward_coef` 默认 `0.0`
- `low_return = reward_int + coef * reward_ext`

## 9. 奖励、成本与指标

### 9.1 环境外在奖励 `reward_ext`

每步每智能体：

- `progress = d_prev - d_curr`
- `r = w_progress * progress - w_time`
- 若到达：`+ reach_bonus`
- 若碰撞：`- collision_penalty`
- 若出界：`- oob_penalty`

### 9.2 内在奖励 `reward_int`

技能定义在 `skills/library.py`，共性项包括：

- 加速度惩罚
- 转向惩罚
- 速度偏差惩罚
- 航向偏差惩罚
- 小权重前进奖励

各技能再附加对应 bonus/penalty（例如 hover 的静止与贴近目标奖励）。

### 9.3 安全成本

- `cost_i = 1` 当步不安全（碰撞或出界），否则 `0`
- 在 `env.info["costs"]` 给出

### 9.4 常用日志

训练/评测常见字段：

- `episode_return_mean`（外在回报均值）
- `safe_reach_ratio`
- `eval_success_rate`, `eval_collision_rate`, `eval_reach_rate`
- `eval_qp_feasible_rate`
- `loss_high_*`, `loss_low_*`

## 10. 默认配置（`default_async_onpolicy_gcbfplus.yaml`）

关键默认值：

- 环境：
  - `n_agents=4`, `n_obstacles=3`, `dt=0.1`, `horizon=200`
  - `action_limit=2.0`, `velocity_limit=2.0`
  - `lidar_beams=32`, `lidar_range=4.5`
  - `neighbor_radius=3.0`
- 同步协议：
  - `mode=async`, `t_sync_max=20`
- 安全：
  - `d_min_agent=0.6`, `d_safe_obs=0.6`
- 技能参数：
  - `ref_speed=1.2`, `decelerate_target_speed=0.3`
  - `cbf_mode=distributed_gcbfplus`
  - `cbf_share_agent=0.5`, `cbf_share_obs=1.0`
  - `use_input_bounds=true`
- 训练：
  - `rollout_steps=200`, `eval_interval=10`
  - `gamma_high=0.99`, `lam_high=0.95`, `gamma_low=0.99`
  - `low_update_mode=target_regression`（可命令行改成 `onpolicy_ppo`）
  - `low_ppo_clip_ratio=0.2`, `low_ppo_value_coef=0.5`, `low_policy_action_std=0.2`

## 11. 训练与评测入口

### 11.1 训练

入口：`python -m hmarl_cbf.train.run_sync_onpolicy`

常用（异步 + GCBF + 下层PPO）：

```bash
python -m hmarl_cbf.train.run_sync_onpolicy \
  --config configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml \
  --low-update-mode onpolicy_ppo \
  --run-name train_async_lowppo_gcbfplus
```

### 11.2 评测

入口：`python -m hmarl_cbf.train.eval_checkpoint_random`

示例：

```bash
python -m hmarl_cbf.train.eval_checkpoint_random \
  --checkpoint artifacts/hmarl_cbf/<run>/checkpoints/last.pt \
  --episodes 100 \
  --seconds 30 \
  --deterministic \
  --run-name eval_100x30s
```

输出：

- `episode_results.csv`
- `summary.json`

## 12. 实现注意事项

1. 世界单位是仿真单位（数值尺度），默认不显式绑定真实米制。
2. 邻居/障碍 CBF 采用“感知到才加约束”：
   - 邻居由 `neighbor_radius` 过滤
   - 障碍由 `lidar_range`（表面距离）过滤
3. 当前版本 QP 不可解时不会强制“减速 fallback”；推理求解器会走内部 stub fallback。
4. 上层回报使用环境外在奖励聚合；下层回报默认以内在奖励为主（`low_ext_reward_coef=0.0`）。

## 13. 一句话总结

当前实现是一个可切换 sync/async 的分层多智能体安全控制框架：上层 MAPPO 学技能切换，下层网络学习 QP 参数并通过 CBF/CLF-QP 逐步生成安全动作，支持 GCBF+ 风格分布式手工 CBF 和下层 on-policy PPO 更新路径。
