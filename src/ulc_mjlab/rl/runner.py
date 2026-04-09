import shutil
from pathlib import Path

import torch
import wandb
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class UlcOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity runner with rolling-latest checkpoint uploads."""

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def _upload_model_mode(self) -> str:
    mode = str(self.cfg.get("upload_model_mode", "rolling_latest"))
    if mode not in {"all", "rolling_latest"}:
      raise ValueError(
        f"Unsupported upload_model_mode `{mode}`. "
        "Expected one of: all, rolling_latest."
      )
    return mode

  def _maybe_upload_checkpoint(self, checkpoint_path: Path) -> None:
    if not self.cfg.get("upload_model", True):
      return

    mode = self._upload_model_mode()
    if mode == "all":
      self.logger.save_model(str(checkpoint_path), self.current_learning_iteration)
      return

    latest_path = checkpoint_path.with_name("model_latest.pt")
    shutil.copy2(checkpoint_path, latest_path)
    self.logger.save_model(str(latest_path), self.current_learning_iteration)

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    self._maybe_upload_checkpoint(Path(path))

    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as exc:
      print(f"[WARN] ONNX export failed (training continues): {exc}")
