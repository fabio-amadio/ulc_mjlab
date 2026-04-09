from dataclasses import dataclass
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg

UploadModelMode = Literal["all", "rolling_latest"]


@dataclass
class UlcOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  upload_model_mode: UploadModelMode = "rolling_latest"
