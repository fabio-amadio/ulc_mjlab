from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import ULCCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SequentialSkillCurriculum:
  """Progressively widens the ULC command space from easy to full-range."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._command = cast(ULCCommand, env.command_manager.get_term(cfg.params["command_name"]))
    self._command_name = cfg.params["command_name"]
    self._metric_names = {
      "velocity": cfg.params["velocity_metric_name"],
      "height": cfg.params["height_metric_name"],
      "upper": cfg.params["upper_metric_name"],
      "torso": cfg.params["torso_metric_name"],
      "hip": cfg.params["hip_metric_name"],
    }
    self._thresholds = {
      "velocity": cfg.params.get("velocity_threshold", 0.8),
      "height": cfg.params.get("height_threshold", 0.85),
      "upper": cfg.params.get("upper_threshold", 0.8),
      "torso": cfg.params.get("torso_threshold", 0.8),
      "hip": cfg.params.get("hip_threshold", 0.2),
    }
    self._increment = cfg.params.get("increment", 0.05)
    self._update_interval_steps = cfg.params.get("update_interval_steps", 1000)

    self._window_sums = {
      name: torch.tensor(0.0, device=env.device) for name in self._metric_names
    }
    self._window_count = 0
    self._last_update_step = 0
    self._last_means = {name: 0.0 for name in self._metric_names}

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    **_kwargs: object,
  ):
    self._command.set_curriculum(
      alpha_height=self._command.alpha_height,
      alpha_upper=self._command.alpha_upper,
    )

    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
      step_count = torch.clamp(
        env.metrics_manager._step_count[env_ids].float(),
        min=1.0,
      )
      for key, metric_name in self._metric_names.items():
        episode_sum = env.metrics_manager._episode_sums[metric_name][env_ids]
        episode_mean = episode_sum / step_count
        self._window_sums[key] += episode_mean.sum()
      self._window_count += len(env_ids)

    step_delta = env.common_step_counter - self._last_update_step
    if step_delta >= self._update_interval_steps and self._window_count > 0:
      for key in self._metric_names:
        self._last_means[key] = (
          self._window_sums[key] / float(self._window_count)
        ).item()

      prev_ready = (
        self._last_means["height"] >= self._thresholds["height"]
        and self._last_means["velocity"] >= self._thresholds["velocity"]
        and self._last_means["hip"] >= self._thresholds["hip"]
      )
      if prev_ready and self._command.alpha_height < 0.98:
        self._command.alpha_height = min(0.98, self._command.alpha_height + self._increment)

      upper_ready = (
        self._last_means["upper"] >= self._thresholds["upper"]
        and self._last_means["torso"] >= self._thresholds["torso"]
        and prev_ready
        and self._command.alpha_height >= 0.98
      )
      if upper_ready and self._command.alpha_upper < 0.98:
        self._command.alpha_upper = min(0.98, self._command.alpha_upper + self._increment)

      for key in self._window_sums:
        self._window_sums[key].zero_()
      self._window_count = 0
      self._last_update_step = env.common_step_counter

    self._command.set_curriculum(
      alpha_height=self._command.alpha_height,
      alpha_upper=self._command.alpha_upper,
    )
    return {
      "alpha_height": self._command.alpha_height,
      "alpha_upper": self._command.alpha_upper,
      "velocity_mean": self._last_means["velocity"],
      "height_mean": self._last_means["height"],
      "upper_mean": self._last_means["upper"],
      "torso_mean": self._last_means["torso"],
      "hip_mean": self._last_means["hip"],
    }
