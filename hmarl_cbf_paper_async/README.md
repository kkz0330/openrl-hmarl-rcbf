# HMARL-CBF Paper-Aligned Async Subset

This folder is a runnable subset of the original `hmarl_cbf` project, copied into an isolated workspace and trimmed to the paper-facing path:

- asynchronous high-level skill updates
- learnable low-level CBF-QP policy
- fixed-parameter distributed CBF-QP baseline
- end-to-end training and checkpoint evaluation

## Main entrypoints

- Variable-parameter HMARL-CBF training:
  - `python -m hmarl_cbf.train.run_async_onpolicy --config configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml`
- Fixed-parameter CBF-QP baseline:
  - `python -m hmarl_cbf.train.run_distributed_cbf_baseline --config configs/hmarl_cbf/baseline_distributed_cbf.yaml`
- Random-scene checkpoint evaluation:
  - `python -m hmarl_cbf.train.eval_checkpoint_random --checkpoint <path-to-last.pt>`
- High-level-skills-only single-UAV simulation:
  - `python -m hmarl_cbf.train.eval_high_level_skills_only --skill-sequence accelerate,turn_left,cruise,decelerate --render-gif`
- High-level-skills-only multi-UAV simulation:
  - `python -m hmarl_cbf.train.eval_high_level_multiagent_skills_only --skill-sequences "accelerate,turn_left,cruise;accelerate,turn_right,cruise" --render-gif`
- High-level-only RL on a fixed single-UAV scene:
  - `python -m hmarl_cbf.train.train_high_level_skills_only --config configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml --total-iterations 200 --episodes-per-iter 16 --eval-interval 20 --eval-episodes 8 --final-render-gif`
- High-level-only RL on a fixed multi-UAV scene:
  - `python -m hmarl_cbf.train.train_high_level_multiagent_skills_only --config configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml --total-iterations 200 --episodes-per-iter 16 --eval-interval 20 --eval-episodes 8 --sync-mode async`

## Minimal dependencies

- Install the runtime dependencies in this folder before launching training or evaluation:
  - `python -m pip install -r requirements.txt`

## Notes

- The async training entrypoint forces:
  - `synchronization.mode = async`
  - `train.low_update_mode = onpolicy_ppo`
- This keeps the copied subset closer to the HMARL-CBF paper than the original mixed engineering branch.
- The baseline is intentionally retained for apples-to-apples comparison against the learned low-level CBF-QP controller.
