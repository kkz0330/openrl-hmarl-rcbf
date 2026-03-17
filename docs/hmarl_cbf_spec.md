# HMARL-CBF On-Policy 实现规范（Step 1）

## 1. 目标与范围
本规范用于把论文第 4/5 节的算法描述映射为可实现的代码接口，场景限定为：
- 2D 多无人机到达任务
- 同时避障（静态障碍）与避碰（其他智能体）
- 逐时间步安全约束
- 先实现同步技能切换版本
- 观测为部分可观测，低层可接入 LiDAR（细节见 `docs/sync_execution_protocol.md`）

本步骤只定义“数学到代码”的一致性，不写训练主循环细节。

## 2. 系统建模（建议采用双积分器）
对智能体 `i` 定义状态与控制：
- `x_i = [p_x, p_y, v_x, v_y]`
- `u_i = [a_x, a_y]`
- 离散动力学 `x_{t+1} = f(x_t, u_t)`（由环境 `dt` 决定）

代码映射：
- `env.obs[i]` 包含局部观测 `o_i`
- `env.state[i]` 对应 `x_i`
- `env.step(action_dict)` 中 `action_dict[i] = u_i`

## 3. 分层策略定义
上层（技能层）：
- 策略 `pi_H(z_i | o_i^H)`，输出离散技能 `z_i`
- 使用 on-policy MAPPO 更新（CTDE）

下层（安全控制层）：
- 输入 `(o_i^L, z_i)` 输出 QP 参数 `theta_qp_i`
- 每个时间步求解参数化 QP 得到 `u_i`
- 通过可微 QP（KKT）对 `theta_qp_i` 反传

代码映射：
- `high_level/actor_critic.py`：`pi_H`, `V_H`
- `low_level/qp_param_net.py`：QP 参数网络
- `low_level/qp_solver.py`：带 CBF/CLF 约束的求解器

## 4. 技能 7 元组映射
每个技能 `k` 定义为：
- 启动集 `I_k`
- 中止集 `T_k`
- 最大持续时间 `tau_k_max`
- 中止函数 `beta_k(o_i, x_env, t_in_skill)`
- 技能安全约束集 `C_k_safe`
- 技能奖励 `r_k_int(s_i, a_i)`
- 安全技能策略 `pi_k_safe`（在本框架里对应“QP 参数化 + CBF/CLF”）

代码映射（建议接口）：
- `skills/base.py::SkillSpec`
- `skills/library.py`：`TurnLeft/TurnRight/Accel/Decel/Cruise`
- `skills/termination.py`：统一 `beta_k` 判定

## 5. CBF-QP 形式（与你的问题直接相关）
结论：可行。  
第一步中把 CBF 设为 GCBF 风格“手工构造 + 分布式局部约束 + 每步 QP 投影”是可行的，而且与“先做可跑通同步版”目标一致。

推荐在 2D 双积分器下使用如下结构：

1) 智能体间安全 CBF（位置域）  
`h_ij = ||p_i - p_j||^2 - d_min^2`

2) 障碍物安全 CBF  
`h_io = ||p_i - p_o||^2 - (r_o + d_safe)^2`

3) 若采用双积分器，优先用高阶/指数 CBF（ECBF）把约束写成对 `u_i` 线性不等式：  
`ddot(h) + k1 * dot(h) + k0 * h >= 0`

4) 分布式实现：  
智能体 `i` 仅对邻居集 `N_i`（通信半径或拓扑图）和局部可见障碍构建约束，不需要全局状态。

## 6. 低层 QP 标准型
每步对每个智能体求解：

`min_{u_i, delta_i} 0.5 * ||u_i - u_ref_i||_R^2 + w_clf * delta_i^2`

约束：
- `A_cbf_i u_i <= b_cbf_i`（硬约束，来自 agent-agent 与 agent-obstacle CBF）
- `A_clf_i u_i <= b_clf_i + delta_i`（软约束）
- `u_min <= u_i <= u_max`
- `delta_i >= 0`

其中：
- `u_ref_i` 可来自技能参考控制器或技能网络
- `A_*, b_*` 由当前局部状态和 QP 参数网络共同决定

代码映射：
- `low_level/constraints.py`：构建 `A_cbf_i, b_cbf_i, A_clf_i, b_clf_i`
- `low_level/controller.py`：`u_ref_i -> QP -> u_i`

## 7. 奖励与优化映射
外在奖励（上层）：
- 到达奖励、时间惩罚、碰撞惩罚、任务完成奖励

内在奖励（下层）：
- 激进加减速惩罚
- 激进转向惩罚
- 偏离参考速度/航向惩罚
- 小权重前进进度奖励

更新方式：
- 上层：MAPPO（on-policy）
- 下层：通过可微 QP 的策略梯度更新 QP 参数网络（可与 on-policy 采样管线对齐）

## 8. 同步版本执行语义（本项目固定）
- 全体智能体在同一决策时刻选技能
- 低层在每个仿真步执行 QP 控制
- 到“全局技能终止时刻”再统一切换下一批技能

这与后续异步版本可兼容，但当前不实现异步。

## 9. Step 1 验收标准
满足以下即通过：
- 所有变量均有“数学符号 -> 代码字段/模块”映射
- CBF/CLF/QP 形式固定且可写成可微求解器输入
- 明确“分布式局部约束”而非全局集中式约束
- 同步执行语义无歧义

## 10. 风险与边界
- 若动力学改成单积分器，CBF 约束公式需降阶，ECBF 写法要同步修改
- 若障碍是非圆形，需要额外支持多面体或 SDF 约束近似
- 分布式局部观测可能造成保守性提升，需要在邻域半径与可行性间权衡
