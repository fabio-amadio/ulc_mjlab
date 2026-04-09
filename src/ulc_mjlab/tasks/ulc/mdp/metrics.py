from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import (
  _DEFAULT_PELVIS_BODY_CFG,
  _DEFAULT_TORSO_BODY_CFG,
  arm_joint_tracking_exp,
  root_height_tracking_exp,
  torso_pitch_tracking_exp,
  torso_roll_tracking_exp,
  torso_yaw_tracking_exp,
  track_angular_velocity_exp,
  track_linear_velocity_exp,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def velocity_tracking_score(
  env: ManagerBasedRlEnv,
  command_name: str,
  std_lin: float,
  std_ang: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  linear = track_linear_velocity_exp(env, command_name, std_lin, asset_cfg)
  angular = track_angular_velocity_exp(env, command_name, std_ang, asset_cfg)
  return 0.5 * (linear + angular)


def height_tracking_score(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return root_height_tracking_exp(env, command_name, std, asset_cfg)


def upper_body_tracking_score(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return arm_joint_tracking_exp(env, command_name, std, asset_cfg)


def torso_tracking_score(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  torso_body_cfg: SceneEntityCfg = _DEFAULT_TORSO_BODY_CFG,
  pelvis_body_cfg: SceneEntityCfg = _DEFAULT_PELVIS_BODY_CFG,
) -> torch.Tensor:
  yaw = torso_yaw_tracking_exp(
    env,
    command_name,
    std,
    asset_cfg,
    torso_body_cfg,
    pelvis_body_cfg,
  )
  roll = torso_roll_tracking_exp(
    env,
    command_name,
    std,
    asset_cfg,
    torso_body_cfg,
    pelvis_body_cfg,
  )
  pitch = torso_pitch_tracking_exp(
    env,
    command_name,
    std,
    asset_cfg,
    torso_body_cfg,
    pelvis_body_cfg,
  )
  return 0.25 * yaw + 0.25 * roll + 0.5 * pitch


class hip_stability_score:
  """Positive stability metric used by the staged curriculum."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._asset_name = cfg.params["asset_cfg"].name
    self._default_joint_pos = asset.data.default_joint_pos
    self._scale = cfg.params.get("scale", 0.3)
    self._minor_ids, _ = asset.find_joints(
      cfg.params["minor_joint_names"],
      preserve_order=True,
    )
    self._major_ids, _ = asset.find_joints(
      cfg.params["major_joint_names"],
      preserve_order=True,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    minor_joint_names: tuple[str, ...] = (),
    major_joint_names: tuple[str, ...] = (),
    scale: float = 0.3,
  ) -> torch.Tensor:
    del asset_cfg, minor_joint_names, major_joint_names, scale
    asset: Entity = env.scene[self._asset_name]
    joint_pos = asset.data.joint_pos
    default_joint_pos = self._default_joint_pos

    minor_error = torch.mean(
      torch.abs(joint_pos[:, self._minor_ids] - default_joint_pos[:, self._minor_ids]),
      dim=1,
    )
    major_error = torch.mean(
      torch.abs(joint_pos[:, self._major_ids] - default_joint_pos[:, self._major_ids]),
      dim=1,
    )
    deviation = minor_error + 2.0 * major_error
    return torch.exp(-deviation / self._scale)
