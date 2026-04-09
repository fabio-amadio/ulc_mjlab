"""ULC-specific observation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import ULCCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def arm_command_abs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(ULCCommand, env.command_manager.get_term(command_name))
  return command.arm_command_abs
