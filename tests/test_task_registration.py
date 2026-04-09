import inspect
import sys
import types
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import ulc_mjlab
from ulc_mjlab.rl import UlcOnPolicyRunner
from ulc_mjlab.utils import (
  get_wandb_checkpoint_path as ulc_get_wandb_checkpoint_path,
)
from ulc_mjlab.utils import (
  get_wandb_curriculum_alphas as ulc_get_wandb_curriculum_alphas,
)


def test_ulc_task_registration() -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  assert task_id in list_tasks()

  env_cfg = load_env_cfg(task_id)
  play_env_cfg = load_env_cfg(task_id, play=True)
  rl_cfg = load_rl_cfg(task_id)

  assert env_cfg.scene.num_envs == 8192
  assert env_cfg.commands["ulc"].resampling_time_range == (1.0, 1.0)
  assert play_env_cfg.scene.num_envs == 1
  assert rl_cfg.max_iterations == 10_000
  assert rl_cfg.upload_model_mode == "rolling_latest"
  assert load_runner_cls(task_id) is UlcOnPolicyRunner


def test_ulc_play_env_zero_action_step() -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  env_cfg = load_env_cfg(task_id, play=True)
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.device)
    env.step(actions)
  finally:
    env.close()


def test_ulc_train_env_step() -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    env.reset()
    actions = torch.zeros(env.action_space.shape, device=env.device)
    env.step(actions)
  finally:
    env.close()


def test_ulc_play_velocity_command_resamples() -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.commands["ulc"].ranges.lin_vel_x = (0.5, 0.5)
  env_cfg.commands["ulc"].ranges.lin_vel_y = (0.0, 0.0)
  env_cfg.commands["ulc"].ranges.ang_vel_z = (0.0, 0.0)
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    env.reset()
    command = env.command_manager.get_term("ulc").velocity_command[0]
    assert torch.allclose(command, torch.tensor([0.5, 0.0, 0.0], device=env.device))
  finally:
    env.close()


def test_ulc_term_params_match_signatures() -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  for play in (False, True):
    env_cfg = load_env_cfg(task_id, play=play)
    term_groups = (
      env_cfg.rewards,
      env_cfg.metrics,
      env_cfg.curriculum,
      env_cfg.terminations,
      env_cfg.events,
    )
    for terms in term_groups:
      for term_cfg in terms.values():
        func = term_cfg.func
        target = func.__call__ if inspect.isclass(func) else func
        signature = inspect.signature(target)
        accepted = set(signature.parameters)
        has_var_kw = any(
          parameter.kind == inspect.Parameter.VAR_KEYWORD
          for parameter in signature.parameters.values()
        )
        if has_var_kw:
          continue
        params = set((term_cfg.params or {}).keys())
        assert params - accepted == set()


def test_ulc_runner_construction(tmp_path: Path) -> None:
  task_id = "Mjlab-ULC-Flat-Unitree-G1"

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 4
  rl_cfg = load_rl_cfg(task_id)
  rl_cfg.logger = "tensorboard"
  rl_cfg.experiment_name = "ulc_test"
  rl_cfg.run_name = "runner_smoke"

  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(rl_cfg), str(tmp_path), "cpu")
    assert runner is not None
  finally:
    env.close()


def test_ulc_patches_mjlab_wandb_checkpoint_resolution() -> None:
  import mjlab.scripts.play as mjlab_play
  import mjlab.scripts.train as mjlab_train
  import mjlab.utils.os as mjlab_os

  assert mjlab_os.get_wandb_checkpoint_path is ulc_get_wandb_checkpoint_path
  assert mjlab_train.get_wandb_checkpoint_path is ulc_get_wandb_checkpoint_path
  assert mjlab_play.get_wandb_checkpoint_path is ulc_get_wandb_checkpoint_path


def test_ulc_patches_play_run_play_for_curriculum_loading() -> None:
  import mjlab.scripts.play as mjlab_play

  assert getattr(mjlab_play, "_ulc_play_curriculum_patched", False) is True
  assert hasattr(mjlab_play.run_play, "__wrapped__")


def test_ulc_wandb_checkpoint_path_supports_model_latest(
  tmp_path: Path, monkeypatch
) -> None:
  class FakeWandbFile:
    def __init__(self, name: str) -> None:
      self.name = name

    def download(self, root: str, replace: bool = True) -> None:
      del replace
      Path(root, self.name).write_text("checkpoint")

  class FakeWandbRun:
    def __init__(self) -> None:
      self._files = [
        FakeWandbFile("events.out"),
        FakeWandbFile("model_latest.pt"),
        FakeWandbFile("policy.onnx"),
      ]

    def files(self) -> list[FakeWandbFile]:
      return self._files

    def file(self, name: str) -> FakeWandbFile:
      return next(file for file in self._files if file.name == name)

  class FakeWandbApi:
    def run(self, run_path: str) -> FakeWandbRun:
      assert run_path == "entity/project/run-123"
      return FakeWandbRun()

  fake_wandb = types.SimpleNamespace(Api=lambda: FakeWandbApi())
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  checkpoint_path, was_cached = ulc_get_wandb_checkpoint_path(
    tmp_path,
    Path("entity/project/run-123"),
  )

  assert checkpoint_path == tmp_path / "wandb_checkpoints" / "run-123" / "model_latest.pt"
  assert checkpoint_path.read_text() == "checkpoint"
  assert was_cached is False


def test_ulc_wandb_curriculum_alpha_resolution(monkeypatch) -> None:
  class FakeWandbRun:
    summary = {
      "Curriculum/skills/alpha_height": 0.35,
      "Curriculum/skills/alpha_upper": 0.10,
    }

  class FakeWandbApi:
    def run(self, run_path: str) -> FakeWandbRun:
      assert run_path == "entity/project/run-123"
      return FakeWandbRun()

  fake_wandb = types.SimpleNamespace(Api=lambda: FakeWandbApi())
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  alpha_height, alpha_upper = ulc_get_wandb_curriculum_alphas(
    "entity/project/run-123"
  )

  assert alpha_height == 0.35
  assert alpha_upper == 0.10


def test_ulc_temporary_play_curriculum_uses_wandb_run_path(monkeypatch) -> None:
  import mjlab.tasks.registry as mjlab_registry

  task_id = "Mjlab-ULC-Flat-Unitree-G1"
  registry_entry = mjlab_registry._REGISTRY[task_id]
  ulc_command = registry_entry.play_env_cfg.commands["ulc"]
  original_alpha_height = ulc_command.alpha_height_init
  original_alpha_upper = ulc_command.alpha_upper_init

  monkeypatch.setattr(ulc_mjlab, "_get_wandb_curriculum_alphas", lambda _: (0.25, 0.15))

  cfg = types.SimpleNamespace(
    wandb_run_path="entity/project/run-123",
  )

  with ulc_mjlab._temporary_ulc_play_curriculum(task_id, cfg):
    assert ulc_command.alpha_height_init == 0.25
    assert ulc_command.alpha_upper_init == 0.15

  assert ulc_command.alpha_height_init == original_alpha_height
  assert ulc_command.alpha_upper_init == original_alpha_upper


def test_ulc_runner_uploads_only_latest_checkpoint(tmp_path: Path) -> None:
  class DummyLogger:
    logger_type = "tensorboard"

    def __init__(self) -> None:
      self.calls: list[tuple[str, int]] = []

    def save_model(self, path: str, iteration: int) -> None:
      self.calls.append((path, iteration))

  checkpoint_path = tmp_path / "model_50.pt"
  checkpoint_path.write_text("weights")

  runner = UlcOnPolicyRunner.__new__(UlcOnPolicyRunner)
  runner.cfg = {"upload_model": True, "upload_model_mode": "rolling_latest"}
  runner.current_learning_iteration = 50
  runner.logger = DummyLogger()

  runner._maybe_upload_checkpoint(checkpoint_path)

  latest_path = tmp_path / "model_latest.pt"
  assert latest_path.read_text() == "weights"
  assert runner.logger.calls == [(str(latest_path), 50)]
