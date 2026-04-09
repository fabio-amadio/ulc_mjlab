from mjlab.tasks.registry import register_mjlab_task

from ulc_mjlab.rl import UlcOnPolicyRunner

from .env_cfgs import unitree_g1_flat_ulc_env_cfg
from .rl_cfg import unitree_g1_ulc_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-ULC-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_ulc_env_cfg(),
  play_env_cfg=unitree_g1_flat_ulc_env_cfg(play=True),
  rl_cfg=unitree_g1_ulc_ppo_runner_cfg(),
  runner_cls=UlcOnPolicyRunner,
)
