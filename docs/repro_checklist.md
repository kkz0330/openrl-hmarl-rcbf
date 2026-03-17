# HMARL-CBF 复现检查清单（Step 12）

## 1. 论文一致性检查（逐项勾选）
- [ ] 任务一致：2D 多智能体到达 + 障碍物 + 智能体间逐步安全约束  
- [ ] 动力学一致：双积分器离散模型（`x=[p_x,p_y,v_x,v_y]`, `u=[a_x,a_y]`）  
- [ ] 观测一致：部分可观测 + LiDAR + 邻居摘要（用于分布式 CBF）  
- [ ] 技能一致：6 技能（左转/右转/加速/减速/巡航/悬停）7 元组完整  
- [ ] 同步语义一致：`sync_switch = all(beta_i) OR max(tau_i) >= T_sync_max`  
- [ ] 低层一致：硬 CBF + 软 CLF 的参数化 QP  
- [ ] 反传一致：可微 QP（KKT 路径）可对低层策略参数求导  
- [ ] 上层一致：on-policy MAPPO（技能策略）  
- [ ] 缓存一致：上层片段级 + 下层步级双缓存；支持 GAE 与 returns  

## 2. 训练配置一致性检查
- [ ] 环境参数固定并版本化（`configs/hmarl_cbf/default_sync_onpolicy.yaml`）  
- [ ] MAPPO 超参数固定（clip/entropy/value_coef/epochs/minibatch）  
- [ ] 低层更新参数固定（`low_update_epochs/max_samples/target_step_scale`）  
- [ ] 所有随机种子记录在实验日志  
- [ ] 每次实验保存 git commit id（若可用）和配置快照  

## 3. 必测指标（每次实验都输出）
- [ ] `success_rate`  
- [ ] `reach_rate`  
- [ ] `collision_rate`  
- [ ] `min_h_agent` / `min_h_obstacle`  
- [ ] `qp_feasible_rate`  
- [ ] `avg_skill_switches`  
- [ ] `avg_episode_return`  

## 4. 基线与消融（建议最小集合）
- Baseline: `full_sync_hmarl_cbf`
- Ablation-A: `no_intrinsic_reward`（下层仅外在奖励）
- Ablation-B: `no_clf_soft`（关闭 CLF soft 约束）
- Ablation-C: `no_lidar`（仅邻居摘要）
- Ablation-D: `single_level`（禁用上层技能，仅低层策略）

## 5. 结果记录要求
- [ ] 每个实验至少 3 个种子  
- [ ] 输出每种子最终指标 + 均值/标准差  
- [ ] 保存训练过程曲线（iteration vs key metrics）  
- [ ] 保存代表性轨迹图或 GIF  

## 6. 复现实验产物目录建议
- `artifacts/repro/{experiment}/{seed}/metrics.json`
- `artifacts/repro/{experiment}/{seed}/train_history.csv`
- `artifacts/repro/repro_summary.csv`
- `artifacts/repro/repro_summary.json`
