"""Base flat-ground ULC task configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp


def make_ulc_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the flat ULC environment configuration."""

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "ulc"},
    ),
  }

  critic_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "ulc"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=6,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=6,
    ),
  }

  metrics = {
    "velocity_tracking": MetricsTermCfg(
      func=mdp.velocity_tracking_score,
      params={"command_name": "ulc", "std_lin": 0.5, "std_ang": 0.5},
    ),
    "height_tracking": MetricsTermCfg(
      func=mdp.height_tracking_score,
      params={"command_name": "ulc", "std": 0.4},
    ),
    "upper_body_tracking": MetricsTermCfg(
      func=mdp.upper_body_tracking_score,
      params={
        "command_name": "ulc",
        "std": 0.35,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "torso_tracking": MetricsTermCfg(
      func=mdp.torso_tracking_score,
      params={
        "command_name": "ulc",
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot"),
        "torso_body_cfg": SceneEntityCfg("robot", body_names=()),
        "pelvis_body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "hip_stability": MetricsTermCfg(
      func=mdp.hip_stability_score,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "minor_joint_names": (),
        "major_joint_names": (),
        "scale": 0.3,
      },
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.UlcJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      arm_joint_names=(),
      command_name="ulc",
      scale=0.25,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "ulc": mdp.ULCCommandCfg(
      entity_name="robot",
      arm_joint_names=(),
      resampling_time_range=(1.0, 1.0),
      debug_vis=False,
      nominal_root_height=0.75,
      upper_body_interval_s=1.0,
      delay_prob=0.5,
      alpha_height_init=0.0,
      alpha_upper_init=0.0,
      ranges=mdp.ULCCommandCfg.Ranges(
        lin_vel_x=(-0.45, 0.55),
        lin_vel_y=(-0.45, 0.45),
        ang_vel_z=(-1.2, 1.2),
        root_height=(0.3, 0.75),
        torso_yaw=(-2.62, 2.62),
        torso_roll=(-0.52, 0.52),
        torso_pitch=(-0.52, 1.57),
      ),
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": {
          0: (0.7, 1.0),
          1: (0.4, 0.7),
          2: (0.0, 0.005),
        },
        "shared_random": True,
      },
    ),
    "payload_and_base_mass": EventTermCfg(
      func=mdp.randomize_payload_and_base_mass,
      mode="reset",
      params={
        "entity_name": "robot",
        "wrist_body_names": ("", ""),
        "base_body_names": (),
        "wrist_mass_range": (0.0, 2.0),
        "base_mass_range": (-5.0, 5.0),
      },
    ),
  }

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity_exp,
      weight=1.0,
      params={"command_name": "ulc", "std": 0.5},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity_exp,
      weight=1.25,
      params={"command_name": "ulc", "std": 0.5},
    ),
    "root_height": RewardTermCfg(
      func=mdp.root_height_tracking_exp,
      weight=1.0,
      params={"command_name": "ulc", "std": 0.4},
    ),
    "upper_body": RewardTermCfg(
      func=mdp.arm_joint_tracking_exp,
      weight=1.0,
      params={
        "command_name": "ulc",
        "std": 0.35,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
      },
    ),
    "torso_yaw": RewardTermCfg(
      func=mdp.torso_yaw_tracking_exp,
      weight=0.25,
      params={
        "command_name": "ulc",
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot"),
        "torso_body_cfg": SceneEntityCfg("robot", body_names=()),
        "pelvis_body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "torso_roll": RewardTermCfg(
      func=mdp.torso_roll_tracking_exp,
      weight=0.25,
      params={
        "command_name": "ulc",
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot"),
        "torso_body_cfg": SceneEntityCfg("robot", body_names=()),
        "pelvis_body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "torso_pitch": RewardTermCfg(
      func=mdp.torso_pitch_tracking_exp,
      weight=0.5,
      params={
        "command_name": "ulc",
        "std": 0.2,
        "asset_cfg": SceneEntityCfg("robot"),
        "torso_body_cfg": SceneEntityCfg("robot", body_names=()),
        "pelvis_body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "center_of_mass": RewardTermCfg(
      func=mdp.center_of_mass_tracking_exp,
      weight=0.5,
      params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "z_velocity": RewardTermCfg(func=mdp.z_linear_velocity_l2, weight=-1.0),
    "energy": RewardTermCfg(func=mdp.mechanical_power_abs, weight=-1.0e-3),
    "joint_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "base_orientation": RewardTermCfg(
      func=mdp.base_orientation_l2_masked,
      weight=-5.0,
      params={"command_name": "ulc"},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_effort_limit": RewardTermCfg(
      func=mdp.joint_effort_limit_cost,
      weight=-2.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),
        "limits": {},
        "soft_ratio": 0.999,
      },
    ),
    "joint_deviation": RewardTermCfg(
      func=mdp.joint_deviation_cost,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "minor_joint_names": (),
        "major_joint_names": (),
        "minor_weight": 0.15,
        "major_weight": 0.3,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_reward,
      weight=0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "ulc",
        "command_threshold": 0.1,
        "cap_s": 0.4,
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide_cost,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "feet_force": RewardTermCfg(
      func=mdp.feet_force_cost,
      weight=-3.0e-3,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble_cost,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "flying": RewardTermCfg(
      func=mdp.flying_cost,
      weight=-1.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contact_cost,
      weight=-1.0,
      params={"sensor_name": "undesired_contacts", "force_threshold": 1.0},
    ),
    "ankle_orientation": RewardTermCfg(
      func=mdp.ankle_orientation_cost,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "root_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2},
    ),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "undesired_contacts", "force_threshold": 10.0},
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  curriculum = {
    "skills": CurriculumTermCfg(
      func=mdp.SequentialSkillCurriculum,
      params={
        "command_name": "ulc",
        "velocity_metric_name": "velocity_tracking",
        "height_metric_name": "height_tracking",
        "upper_metric_name": "upper_body_tracking",
        "torso_metric_name": "torso_tracking",
        "hip_metric_name": "hip_stability",
        "velocity_threshold": 0.8,
        "height_threshold": 0.85,
        "upper_threshold": 0.8,
        "torso_threshold": 0.8,
        "hip_threshold": 0.2,
        "increment": 0.05,
        "update_interval_steps": 1000,
      },
    )
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(),
      num_envs=8192,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      contact_sensor_maxmatch=64,
      nconmax=None,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
