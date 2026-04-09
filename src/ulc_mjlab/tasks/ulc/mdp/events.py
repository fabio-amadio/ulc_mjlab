from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def randomize_payload_and_base_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  wrist_body_names: tuple[str, str],
  base_body_names: tuple[str, ...],
  wrist_mass_range: tuple[float, float],
  base_mass_range: tuple[float, float],
  entity_name: str = "robot",
) -> None:
  """Randomize per-episode wrist payloads and torso mass bias."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  asset = env.scene[entity_name]
  wrist_ids, _ = asset.find_bodies(wrist_body_names, preserve_order=True)
  base_ids, _ = asset.find_bodies(base_body_names, preserve_order=True)

  wrist_global_ids = torch.as_tensor(
    asset.indexing.body_ids[wrist_ids],
    device=env.device,
    dtype=torch.long,
  )
  base_global_ids = torch.as_tensor(
    asset.indexing.body_ids[base_ids],
    device=env.device,
    dtype=torch.long,
  )

  if len(wrist_global_ids) != 2:
    raise ValueError("ULC payload randomization expects exactly two wrist bodies.")

  body_mass = env.sim.model.body_mass
  default_mass = env.sim.get_default_field("body_mass")

  env_grid, wrist_grid = torch.meshgrid(env_ids, wrist_global_ids, indexing="ij")
  wrist_default = default_mass[wrist_global_ids].unsqueeze(0).repeat(len(env_ids), 1)
  total_payload = torch.empty(len(env_ids), device=env.device).uniform_(*wrist_mass_range)
  split = torch.rand(len(env_ids), device=env.device)
  payload = torch.stack([total_payload * split, total_payload * (1.0 - split)], dim=-1)
  body_mass[env_grid, wrist_grid] = torch.clamp(wrist_default + payload, min=1.0e-5)

  if len(base_global_ids) > 0:
    env_grid, base_grid = torch.meshgrid(env_ids, base_global_ids, indexing="ij")
    base_default = default_mass[base_global_ids].unsqueeze(0).repeat(len(env_ids), 1)
    base_delta = torch.empty(len(env_ids), device=env.device).uniform_(*base_mass_range)
    body_mass[env_grid, base_grid] = torch.clamp(
      base_default + base_delta.unsqueeze(-1) / float(len(base_global_ids)),
      min=1.0e-5,
    )
