from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _quintic_smooth(progress: torch.Tensor) -> torch.Tensor:
  return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


@dataclass(kw_only=True)
class ULCCommandCfg(CommandTermCfg):
  entity_name: str
  arm_joint_names: tuple[str, ...]
  nominal_root_height: float = 0.75
  upper_body_interval_s: float = 1.0
  delay_prob: float = 0.5
  alpha_height_init: float = 0.0
  alpha_upper_init: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    root_height: tuple[float, float]
    torso_yaw: tuple[float, float]
    torso_roll: tuple[float, float]
    torso_pitch: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ULCCommand:
    return ULCCommand(self, env)


class ULCCommand(CommandTerm):
  """Unified command generator for velocity, height, torso, and arm goals."""

  cfg: ULCCommandCfg

  def __init__(self, cfg: ULCCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    arm_joint_ids, _ = self.robot.find_joints(cfg.arm_joint_names, preserve_order=True)
    self.arm_joint_ids = torch.tensor(
      arm_joint_ids,
      device=self.device,
      dtype=torch.long,
    )
    self.default_arm_joint_pos = self.robot.data.default_joint_pos[
      :, self.arm_joint_ids
    ].clone()
    self.arm_joint_limits = self.robot.data.joint_pos_limits[
      :, self.arm_joint_ids
    ].clone()

    self.alpha_height = cfg.alpha_height_init
    self.alpha_upper = cfg.alpha_upper_init
    self._last_dt = 0.0
    self._interp_steps = max(1, int(round(cfg.upper_body_interval_s / env.step_dt)))

    self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
    self.height_command = torch.full(
      (self.num_envs, 1),
      cfg.nominal_root_height,
      device=self.device,
    )
    self.torso_command = torch.zeros(self.num_envs, 3, device=self.device)
    self.arm_command_rel = torch.zeros(
      self.num_envs,
      len(arm_joint_ids),
      device=self.device,
    )

    self._height_start = self.height_command.clone()
    self._height_goal = self.height_command.clone()
    self._torso_start = self.torso_command.clone()
    self._torso_goal = self.torso_command.clone()
    self._arm_start_rel = self.arm_command_rel.clone()
    self._arm_goal_rel = self.arm_command_rel.clone()
    self._arm_theoretical_prev_rel = self.arm_command_rel.clone()
    self._arm_delay_accum_rel = self.arm_command_rel.clone()
    self._interp_step = torch.zeros(
      self.num_envs,
      device=self.device,
      dtype=torch.long,
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.velocity_command,
        self.height_command,
        self.torso_command,
        self.arm_command_abs,
      ],
      dim=-1,
    )

  @property
  def arm_command_abs(self) -> torch.Tensor:
    return self.default_arm_joint_pos + self.arm_command_rel

  def set_curriculum(self, *, alpha_height: float, alpha_upper: float) -> None:
    self.alpha_height = float(alpha_height)
    self.alpha_upper = float(alpha_upper)

  def compute(self, dt: float) -> None:
    self._last_dt = dt
    super().compute(dt)

  def _update_metrics(self) -> None:
    return

  def _sample_height_goal(self, env_ids: torch.Tensor) -> torch.Tensor:
    sampled = torch.empty((len(env_ids), 1), device=self.device)
    sampled.uniform_(*self.cfg.ranges.root_height)
    return self.cfg.nominal_root_height + self.alpha_height * (
      sampled - self.cfg.nominal_root_height
    )

  def _sample_torso_goal(self, env_ids: torch.Tensor) -> torch.Tensor:
    goal = torch.empty((len(env_ids), 3), device=self.device)
    goal[:, 0].uniform_(*self.cfg.ranges.torso_yaw)
    goal[:, 1].uniform_(*self.cfg.ranges.torso_roll)
    goal[:, 2].uniform_(*self.cfg.ranges.torso_pitch)
    return goal * self.alpha_upper

  def _sample_arm_goal_rel(self, env_ids: torch.Tensor) -> torch.Tensor:
    lower = self.arm_joint_limits[env_ids, :, 0]
    upper = self.arm_joint_limits[env_ids, :, 1]
    sample = lower + torch.rand_like(lower) * (upper - lower)
    return self.alpha_upper * (sample - self.default_arm_joint_pos[env_ids])

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.velocity_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.velocity_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.velocity_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    self._height_start[env_ids] = self.height_command[env_ids]
    self._height_goal[env_ids] = self._sample_height_goal(env_ids)

    self._torso_start[env_ids] = self.torso_command[env_ids]
    self._torso_goal[env_ids] = self._sample_torso_goal(env_ids)

    self._arm_start_rel[env_ids] = self.arm_command_rel[env_ids]
    self._arm_goal_rel[env_ids] = self._sample_arm_goal_rel(env_ids)
    self._arm_theoretical_prev_rel[env_ids] = self.arm_command_rel[env_ids]
    self._arm_delay_accum_rel[env_ids] = 0.0
    self._interp_step[env_ids] = 0

  def _update_command(self) -> None:
    if self._last_dt > 0.0:
      self._interp_step += 1

    progress = torch.clamp(
      self._interp_step.float() / float(self._interp_steps),
      min=0.0,
      max=1.0,
    ).unsqueeze(-1)
    smooth = _quintic_smooth(progress)

    self.height_command = self._height_start + (
      self._height_goal - self._height_start
    ) * smooth
    self.torso_command = self._torso_start + (
      self._torso_goal - self._torso_start
    ) * smooth

    theoretical_arm_rel = self._arm_start_rel + (
      self._arm_goal_rel - self._arm_start_rel
    ) * smooth

    if self._last_dt <= 0.0:
      return

    delta = theoretical_arm_rel - self._arm_theoretical_prev_rel
    delay_mask = (
      torch.rand_like(delta) < self.cfg.delay_prob
    ).to(dtype=delta.dtype)
    release_mask = 1.0 - delay_mask

    effective_delta = (
      delta * release_mask + self._arm_delay_accum_rel * release_mask
    )
    self.arm_command_rel += effective_delta
    self._arm_delay_accum_rel = (
      self._arm_delay_accum_rel * delay_mask + delta * delay_mask
    )
    self._arm_theoretical_prev_rel = theoretical_arm_rel

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity arrows plus height markers."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    cmd_vel_bs = self.velocity_command.cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset
    height_sphere_radius = max(0.02, 0.04 * float(visualizer.meansize))
    height_marker_offset_b = np.array([-0.10, 0.0, 0.0], dtype=np.float64)

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd_vel_b = cmd_vel_bs[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      cmd_lin_from = local_to_world(np.array([0.0, 0.0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([cmd_vel_b[0], cmd_vel_b[1], 0.0]))
        * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, cmd_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      act_lin_from = cmd_lin_from
      act_lin_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0.0]))
        * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      act_ang_from = cmd_lin_from
      act_ang_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )

      height_marker_base_pos = local_to_world(height_marker_offset_b)

      current_height_pos = height_marker_base_pos.copy()
      current_height_pos[2] = float(base_pos_w[2])
      visualizer.add_sphere(
        current_height_pos,
        radius=height_sphere_radius,
        color=(0.0, 0.9, 1.0, 0.9),
      )

      cmd_height_pos = height_marker_base_pos.copy()
      cmd_height_pos[2] = float(self.height_command[batch, 0])
      visualizer.add_sphere(
        cmd_height_pos,
        radius=height_sphere_radius,
        color=(1.0, 0.55, 0.1, 0.85),
      )
