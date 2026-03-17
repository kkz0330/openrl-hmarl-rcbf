# HMARL-CBF 同步执行协议（Step 2）

## 1. 目标
定义“同步版本”分层控制的唯一执行语义，避免后续实现出现时序歧义。  
本协议同时纳入 LiDAR 部分可观测设定，并保持与 GCBF 风格局部/分布式安全控制兼容。

## 2. 时间轴与同步切换规则
设环境仿真步为 `t = 0, 1, 2, ...`，上层技能决策轮次为 `k = 0, 1, 2, ...`。

- `t_k`：第 `k` 轮上层统一决策时刻
- 在 `t_k` 时，所有智能体同时采样技能 `z_i^k ~ pi_H(.|o_i^H(t_k))`
- 在区间 `[t_k, t_{k+1})` 内，所有智能体仅由下层控制器逐步输出动作
- `t_{k+1}` 触发条件（同步版固定）：
  - 全体智能体都满足各自技能终止条件，或
  - 达到本轮全局上限 `T_sync_max`（防止死锁）

说明：
- 同步版不允许“个别智能体提前换技能”。
- 若单个体已满足终止条件但未到 `t_{k+1}`，其技能保持“已终止但冻结”，下层继续执行安全保持策略直到全局切换点。

## 3. 每个仿真步的数据流
在每个 `t`（`t_k <= t < t_{k+1}`）按下列顺序执行：

1. 局部观测更新：`o_i^L(t)`（含 LiDAR 与邻居信息）
2. 低层网络前向：`theta_qp_i(t) = f_L(o_i^L(t), z_i^k)`
3. 构建约束：`A_cbf_i, b_cbf_i, A_clf_i, b_clf_i`
4. 求解 QP：`u_i(t) = argmin QP_i(...)`
5. 环境推进：`x(t+1) = F(x(t), u(t))`
6. 记录 transition（低层步级）
7. 更新该智能体技能内部计时器 `tau_i <- tau_i + 1`

## 4. 上层轮次数据聚合（on-policy）
在 `t_{k+1}` 时，对每个智能体聚合本轮片段：

- 片段起点：`t_k`
- 片段终点：`t_{k+1}`
- 技能：`z_i^k`
- 外在回报：`R_i^H(k) = sum_{t=t_k}^{t_{k+1}-1} gamma_H^(t-t_k) * r_i^ext(t)`
- 终止标记：`done_i^H(k)`（到达/碰撞/超时/episode 结束）

该片段作为 MAPPO 的一条上层样本。

## 5. LiDAR 部分可观测协议
每个智能体 `i` 的低层观测 `o_i^L` 由以下组成：

- 自身状态：`x_i`（位置、速度、航向/角速度，按动力学定义）
- 目标相对量：`goal_i - p_i`、参考速度等
- LiDAR 向量：`scan_i in R^{N_beam}`
- 可选邻居摘要：最近邻相对位姿/速度（用于分布式 CBF 约束构建）

LiDAR 计算约定：

- 扇区数：`N_beam`（配置项）
- 最大量程：`R_lidar`（配置项）
- 返回值：每束最小碰撞距离，未命中返回 `R_lidar`
- 归一化：`scan_i_norm = scan_i / R_lidar`
- 可选噪声：`epsilon ~ N(0, sigma_lidar^2)`（训练可开，评估默认关）

与 GCBF 风格兼容性：

- 不冲突。  
- GCBF 的核心是局部图信息与分布式安全约束；LiDAR 可作为局部障碍感知来源。
- 对“其他智能体避碰”建议仍保留显式邻居状态（或可由通信图提供），不要只依赖 LiDAR 点云，以确保 CBF 约束可稳定写成解析形式。

## 6. CBF 约束构建的观测来源（固定）
为了保证可解释与可微分：

- agent-agent CBF：来自邻居相对状态（通信图/邻域查询）
- agent-obstacle CBF：来自障碍几何先验 + LiDAR 命中信息辅助

不采用“仅凭 LiDAR 端到端学 CBF 约束”的黑盒方式。

## 7. 同步终止判定函数（统一接口）
定义本轮全局切换判定：

`sync_switch = all_i(beta_i == 1) OR (max_i(tau_i) >= T_sync_max)`

其中每个体终止函数：

`beta_i = 1 if (x_i in T_{z_i^k}) OR (tau_i >= tau_{z_i^k,max}) else 0`

## 8. 必要配置项（Step 2 固定）
- `dt`
- `T_sync_max`
- `N_beam`, `R_lidar`, `sigma_lidar`
- `neighbor_radius` 或 `topology_k`
- `d_min_agent`, `d_safe_obs`
- `u_min`, `u_max`
- `episode_horizon`

## 9. 日志字段（用于后续调试）
每步记录：
- `t, agent_id, z_i^k, tau_i`
- `qp_feasible, qp_iter, qp_obj`
- `min_h_agent, min_h_obs`
- `lidar_min, lidar_mean`
- `collision_flag, reach_flag`

每轮（上层）记录：
- `k, t_k, t_{k+1}, skill_duration`
- `R_i^H, advantage_i^H, value_i^H`

## 10. Step 2 验收标准
- 同步切换条件唯一且可代码化（见第 7 节）
- 上下层样本边界清晰（步级 vs 轮次级）
- LiDAR 接入低层观测且不破坏分布式 CBF 构建
- 配置项与日志字段足以支撑第 3/4/5 步开发
