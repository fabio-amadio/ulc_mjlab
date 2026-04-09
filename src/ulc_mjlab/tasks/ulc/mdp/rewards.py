from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .commands import ULCCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_TORSO_BODY_CFG = SceneEntityCfg("robot", body_names=("torso_link",))
_DEFAULT_PELVIS_BODY_CFG = SceneEntityCfg("robot", body_names=("pelvis",))


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> ULCCommand:
  return cast(ULCCommand, env.command_manager.get_term(command_name))


def _single_body_index(body_cfg: SceneEntityCfg) -> int:
  if isinstance(body_cfg.body_ids, slice):
    raise ValueError("Expected a single resolved body id, got slice(None).")
  if len(body_cfg.body_ids) != 1:
    raise ValueError(f"Expected exactly one body, got {body_cfg.body_names}.")
  return body_cfg.body_ids[0]


def _joint_ids_and_matching_actuator_ids(
  asset: Entity,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor | slice, list[int]]:
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    joint_names = tuple(asset.joint_names)
  else:
    joint_names = tuple(asset.joint_names[joint_id] for joint_id in joint_ids)

  actuator_ids, actuator_names = asset.find_actuators(joint_names, preserve_order=True)
  if tuple(actuator_names) != joint_names:
    raise ValueError(
      "Expected a one-to-one mapping between joint names and actuator names "
      f"for {joint_names}, got {tuple(actuator_names)}."
    )

  return joint_ids, actuator_ids


def _torso_relative_zxy(
  asset: Entity,
  torso_body_cfg: SceneEntityCfg,
  pelvis_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  torso_body_id = _single_body_index(torso_body_cfg)
  pelvis_body_id = _single_body_index(pelvis_body_cfg)

  torso_quat_w = asset.data.body_link_quat_w[:, torso_body_id, :]
  pelvis_quat_w = asset.data.body_link_quat_w[:, pelvis_body_id, :]
  torso_quat_rel = quat_mul(quat_conjugate(pelvis_quat_w), torso_quat_w)
  torso_rot_rel = matrix_from_quat(torso_quat_rel)

  yaw = torch.atan2(-torso_rot_rel[:, 0, 1], torso_rot_rel[:, 1, 1])
  roll = torch.asin(torch.clamp(torso_rot_rel[:, 2, 1], min=-1.0, max=1.0))
  pitch = torch.atan2(-torso_rot_rel[:, 2, 0], torso_rot_rel[:, 2, 2])
  return torch.stack([yaw, roll, pitch], dim=-1)


def track_linear_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  error = torch.sum(
    torch.square(command.velocity_command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
    dim=1,
  )
  return torch.exp(-error / std**2)


def track_angular_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  error = torch.square(
    command.velocity_command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
  )
  return torch.exp(-error / std**2)


def root_height_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  error = torch.square(
    asset.data.root_link_pos_w[:, 2] - command.height_command.squeeze(-1)
  )
  return torch.exp(-error / std**2)


def arm_joint_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  error = torch.sum(torch.square(joint_pos - command.arm_command_abs), dim=1)
  return torch.exp(-error / std**2)


def torso_yaw_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  torso_body_cfg: SceneEntityCfg = _DEFAULT_TORSO_BODY_CFG,
  pelvis_body_cfg: SceneEntityCfg = _DEFAULT_PELVIS_BODY_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  torso_angles = _torso_relative_zxy(asset, torso_body_cfg, pelvis_body_cfg)
  error = torch.square(torso_angles[:, 0] - command.torso_command[:, 0])
  return torch.exp(-error / std**2)


def torso_roll_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  torso_body_cfg: SceneEntityCfg = _DEFAULT_TORSO_BODY_CFG,
  pelvis_body_cfg: SceneEntityCfg = _DEFAULT_PELVIS_BODY_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  torso_angles = _torso_relative_zxy(asset, torso_body_cfg, pelvis_body_cfg)
  error = torch.square(torso_angles[:, 1] - command.torso_command[:, 1])
  return torch.exp(-error / std**2)


def torso_pitch_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  torso_body_cfg: SceneEntityCfg = _DEFAULT_TORSO_BODY_CFG,
  pelvis_body_cfg: SceneEntityCfg = _DEFAULT_PELVIS_BODY_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  torso_angles = _torso_relative_zxy(asset, torso_body_cfg, pelvis_body_cfg)
  error = torch.square(torso_angles[:, 2] - command.torso_command[:, 2])
  return torch.exp(-error / std**2)


def center_of_mass_tracking_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  com_xy = asset.data.root_com_pos_w[:, :2]
  feet_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2].mean(dim=1)
  error = torch.sum(torch.square(com_xy - feet_xy), dim=1)
  return torch.exp(-error / std**2)


def z_linear_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def mechanical_power_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids, actuator_ids = _joint_ids_and_matching_actuator_ids(asset, asset_cfg)
  torque = asset.data.actuator_force[:, actuator_ids]
  velocity = asset.data.joint_vel[:, joint_ids]
  return torch.sum(torch.abs(torque * velocity), dim=1)


def base_orientation_l2_masked(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _get_command(env, command_name)
  base_roll, base_pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  roll_limit = max(
    abs(command.cfg.ranges.torso_roll[0]),
    abs(command.cfg.ranges.torso_roll[1]),
    1.0e-6,
  )
  pitch_limit = max(
    abs(command.cfg.ranges.torso_pitch[0]),
    abs(command.cfg.ranges.torso_pitch[1]),
    1.0e-6,
  )
  mask_roll = torch.clamp(1.0 - torch.abs(command.torso_command[:, 1]) / roll_limit, 0.0, 1.0)
  mask_pitch = torch.clamp(
    1.0 - torch.abs(command.torso_command[:, 2]) / pitch_limit,
    0.0,
    1.0,
  )
  return base_roll.square() * mask_roll + base_pitch.square() * mask_pitch


class joint_effort_limit_cost:
  """Penalize waist torques that exceed a configured soft effort limit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._asset_name = cfg.params["asset_cfg"].name
    joint_ids, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
      preserve_order=True,
    )
    actuator_ids, actuator_names = asset.find_actuators(
      joint_names,
      preserve_order=True,
    )
    if tuple(actuator_names) != tuple(joint_names):
      raise ValueError(
        "Expected waist actuator names to match joint names, "
        f"got joints={tuple(joint_names)} and actuators={tuple(actuator_names)}."
      )
    _, _, limits = resolve_matching_names_values(
      cfg.params["limits"],
      joint_names,
    )
    self._joint_ids = joint_ids
    self._actuator_ids = actuator_ids
    self._limits = torch.tensor(limits, device=env.device, dtype=torch.float32)
    self._soft_ratio = cfg.params.get("soft_ratio", 0.999)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    limits: dict[str, float] | None = None,
    soft_ratio: float = 0.999,
  ) -> torch.Tensor:
    del asset_cfg, limits, soft_ratio
    asset: Entity = env.scene[self._asset_name]
    torque = torch.abs(asset.data.actuator_force[:, self._actuator_ids])
    excess = torch.clamp(torque - self._soft_ratio * self._limits, min=0.0)
    return torch.sum(excess, dim=1)


class joint_deviation_cost:
  """Penalize selected hip and ankle joints deviating from default posture."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._default_joint_pos = asset.data.default_joint_pos
    self._asset_name = cfg.params["asset_cfg"].name
    self._minor_weight = cfg.params.get("minor_weight", 0.15)
    self._major_weight = cfg.params.get("major_weight", 0.3)
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
    minor_weight: float = 0.15,
    major_weight: float = 0.3,
  ) -> torch.Tensor:
    del asset_cfg, minor_joint_names, major_joint_names, minor_weight, major_weight
    asset: Entity = env.scene[self._asset_name]
    joint_pos = asset.data.joint_pos
    default_joint_pos = self._default_joint_pos

    minor_error = torch.sum(
      torch.abs(joint_pos[:, self._minor_ids] - default_joint_pos[:, self._minor_ids]),
      dim=1,
    )
    major_error = torch.sum(
      torch.abs(joint_pos[:, self._major_ids] - default_joint_pos[:, self._major_ids]),
      dim=1,
    )
    return self._minor_weight * minor_error + self._major_weight * major_error


def feet_air_time_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
  cap_s: float = 0.4,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  command = _get_command(env, command_name)

  assert sensor.data.found is not None
  assert sensor.data.last_air_time is not None

  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  single_stance = torch.sum((sensor.data.found > 0).float(), dim=1) == 1
  active = torch.norm(command.velocity_command[:, :2], dim=1) > command_threshold
  reward = torch.sum(torch.minimum(sensor.data.last_air_time, torch.full_like(sensor.data.last_air_time, cap_s)) * first_contact.float(), dim=1)
  return reward * single_stance.float() * active.float()


def feet_slide_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  assert sensor.data.found is not None
  contact = (sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return torch.sum(torch.norm(foot_vel_xy, dim=-1) * contact, dim=1)


def feet_force_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_z = torch.abs(sensor.data.force[:, :, 2])
  return torch.sum(torch.clamp(force_z - 500.0, min=0.0, max=400.0), dim=1)


def feet_stumble_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_xy = torch.norm(sensor.data.force[:, :, :2], dim=-1)
  force_z = torch.abs(sensor.data.force[:, :, 2])
  return torch.sum((force_xy > (5.0 * force_z)).float(), dim=1)


def flying_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (torch.sum(sensor.data.found > 0, dim=1) == 0).float()


def undesired_contact_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()
  if data.force is not None:
    force_mag = torch.norm(data.force, dim=-1)
    return (force_mag > force_threshold).sum(dim=-1).float()
  assert data.found is not None
  return (data.found > 0).sum(dim=-1).float()


def ankle_orientation_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  gravity_w = asset.data.gravity_vec_w.unsqueeze(1).expand_as(body_quat_w[..., :3])
  projected_gravity = quat_apply_inverse(body_quat_w, gravity_w)
  return torch.sum(torch.sum(torch.square(projected_gravity[..., :2]), dim=-1), dim=1)
