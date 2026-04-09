from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .commands import ULCCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class UlcJointPositionActionCfg(BaseActionCfg):
  """Joint position targets with residual arm commands around ULC arm goals."""

  arm_joint_names: tuple[str, ...] | list[str]
  command_name: str
  use_default_offset: bool = True

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> UlcJointPositionAction:
    return UlcJointPositionAction(self, env)


class UlcJointPositionAction(BaseAction):
  """Apply default-offset actions, except for arms which use command residuals."""

  cfg: UlcJointPositionActionCfg

  def __init__(self, cfg: UlcJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self._command = cast(ULCCommand, env.command_manager.get_term(cfg.command_name))

    arm_joint_ids, _ = self._entity.find_joints(
      cfg.arm_joint_names,
      preserve_order=True,
    )
    arm_joint_id_to_index = {joint_id: i for i, joint_id in enumerate(arm_joint_ids)}

    arm_mask = [
      int(target_id) in arm_joint_id_to_index for target_id in self.target_ids.tolist()
    ]
    self._arm_mask = torch.tensor(arm_mask, device=self.device, dtype=torch.bool)
    self._non_arm_mask = ~self._arm_mask
    self._arm_command_order = torch.tensor(
      [
        arm_joint_id_to_index[int(target_id)]
        for target_id in self.target_ids[self._arm_mask].tolist()
      ],
      device=self.device,
      dtype=torch.long,
    )

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self.target_ids].clone()

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    scaled_actions = self._raw_actions * self._scale
    self._processed_actions = scaled_actions + self._offset

    if self._arm_mask.any():
      arm_targets = self._command.arm_command_abs[:, self._arm_command_order]
      self._processed_actions[:, self._arm_mask] = (
        scaled_actions[:, self._arm_mask] + arm_targets
      )

    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self.target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self.target_ids)
