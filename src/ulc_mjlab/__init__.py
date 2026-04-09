"""Standalone ULC task package built on top of mjlab."""

from contextlib import contextmanager
from pathlib import Path

from ulc_mjlab.utils import (
  get_wandb_checkpoint_path as _get_wandb_checkpoint_path,
)
from ulc_mjlab.utils import (
  get_wandb_curriculum_alphas as _get_wandb_curriculum_alphas,
)


def _patch_mjlab_wandb_checkpoint_loading() -> None:
  """Teach upstream mjlab CLIs how to resolve rolling-latest W&B checkpoints."""
  try:
    import mjlab.utils.os as mjlab_os

    mjlab_os.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
  except Exception:
    pass

  for module_name in (
    "mjlab.scripts.train",
    "mjlab.scripts.play",
    "mjlab.tasks.tracking.scripts.evaluate",
  ):
    try:
      module = __import__(module_name, fromlist=["get_wandb_checkpoint_path"])
      module.get_wandb_checkpoint_path = _get_wandb_checkpoint_path
    except Exception:
      pass


def _resolve_ulc_play_curriculum_alphas(
  cfg,
) -> tuple[float | None, float | None]:
  wandb_run_path = getattr(cfg, "wandb_run_path", None)
  if wandb_run_path is None:
    return None, None
  return _get_wandb_curriculum_alphas(wandb_run_path)


@contextmanager
def _temporary_ulc_play_curriculum(task_id: str, cfg):
  if task_id != "Mjlab-ULC-Flat-Unitree-G1":
    yield
    return

  import mjlab.tasks.registry as mjlab_registry

  registry_entry = mjlab_registry._REGISTRY.get(task_id)
  if registry_entry is None:
    yield
    return

  ulc_command = registry_entry.play_env_cfg.commands.get("ulc")
  if ulc_command is None:
    yield
    return

  original_alpha_height = ulc_command.alpha_height_init
  original_alpha_upper = ulc_command.alpha_upper_init

  alpha_height, alpha_upper = _resolve_ulc_play_curriculum_alphas(cfg)

  try:
    if alpha_height is not None:
      ulc_command.alpha_height_init = float(alpha_height)
    if alpha_upper is not None:
      ulc_command.alpha_upper_init = float(alpha_upper)

    if alpha_height is not None or alpha_upper is not None:
      print(
        "[INFO]: Play curriculum overrides active: "
        f"alpha_height={ulc_command.alpha_height_init:.3f}, "
        f"alpha_upper={ulc_command.alpha_upper_init:.3f}"
      )
    yield
  finally:
    ulc_command.alpha_height_init = original_alpha_height
    ulc_command.alpha_upper_init = original_alpha_upper


def _patch_mjlab_play_curriculum_controls() -> None:
  try:
    import mjlab.scripts.play as mjlab_play
  except Exception:
    return

  if getattr(mjlab_play, "_ulc_play_curriculum_patched", False):
    return

  original_run_play = mjlab_play.run_play

  def patched_run_play(task_id: str, cfg):
    with _temporary_ulc_play_curriculum(task_id, cfg):
      return original_run_play(task_id, cfg)

  patched_run_play.__wrapped__ = original_run_play
  mjlab_play.run_play = patched_run_play
  mjlab_play._ulc_play_curriculum_patched = True


_patch_mjlab_wandb_checkpoint_loading()
_patch_mjlab_play_curriculum_controls()

ULC_MJLAB_SRC_PATH = Path(__file__).resolve().parent
