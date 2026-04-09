"""Standalone ULC task package built on top of mjlab."""

from pathlib import Path

from ulc_mjlab.utils import (
  get_wandb_checkpoint_path as _get_wandb_checkpoint_path,
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


_patch_mjlab_wandb_checkpoint_loading()

ULC_MJLAB_SRC_PATH = Path(__file__).resolve().parent
