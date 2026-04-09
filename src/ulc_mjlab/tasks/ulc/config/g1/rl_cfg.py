"""RSL-RL configuration for Unitree G1 flat ULC training."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlPpoAlgorithmCfg,
)

from ulc_mjlab.rl import UlcOnPolicyRunnerCfg


def unitree_g1_ulc_ppo_runner_cfg() -> UlcOnPolicyRunnerCfg:
  """Create the PPO runner configuration used by the ULC paper."""

  return UlcOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(1024, 512, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 512, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.006,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_ulc_flat",
    num_steps_per_env=24,
    max_iterations=10_000,
    save_interval=50,
    wandb_tags=("ulc", "g1", "flat"),
  )
