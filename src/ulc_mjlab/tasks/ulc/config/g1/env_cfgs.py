"""Unitree G1 flat ULC environment configuration."""

from copy import deepcopy

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION_WITHOUT_SELF
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ulc_mjlab.tasks.ulc.mdp.actions import UlcJointPositionActionCfg
from ulc_mjlab.tasks.ulc.mdp.commands import ULCCommandCfg
from ulc_mjlab.tasks.ulc.ulc_env_cfg import make_ulc_env_cfg

ARM_JOINT_NAMES = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
FOOT_SITE_NAMES = ("left_foot", "right_foot")
ANKLE_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_BODY_NAME = "torso_link"
PELVIS_BODY_NAME = "pelvis"
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
HIP_MINOR_JOINT_NAMES = (
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
)
HIP_MAJOR_JOINT_NAMES = ("left_hip_roll_joint", "right_hip_roll_joint")
WAIST_EFFORT_LIMITS = {
  "waist_yaw_joint": 88.0,
  "waist_roll_joint": 50.0,
  "waist_pitch_joint": 50.0,
}
G1_KP_OVERRIDES = {
  ".*_hip_yaw_joint": 100.0,
  ".*_hip_roll_joint": 100.0,
  ".*_hip_pitch_joint": 100.0,
  ".*_knee_joint": 200.0,
  ".*_ankle_pitch_joint": 20.0,
  ".*_ankle_roll_joint": 20.0,
  "waist_yaw_joint": 300.0,
  "waist_roll_joint": 300.0,
  "waist_pitch_joint": 300.0,
  ".*_shoulder_pitch_joint": 90.0,
  ".*_shoulder_roll_joint": 60.0,
  ".*_shoulder_yaw_joint": 20.0,
  ".*_elbow_joint": 60.0,
  ".*_wrist_roll_joint": 4.0,
  ".*_wrist_pitch_joint": 4.0,
  ".*_wrist_yaw_joint": 4.0,
}
G1_KD_OVERRIDES = {
  ".*_hip_yaw_joint": 2.5,
  ".*_hip_roll_joint": 2.5,
  ".*_hip_pitch_joint": 2.5,
  ".*_knee_joint": 5.0,
  ".*_ankle_pitch_joint": 0.2,
  ".*_ankle_roll_joint": 0.1,
  "waist_yaw_joint": 5.0,
  "waist_roll_joint": 5.0,
  "waist_pitch_joint": 5.0,
  ".*_shoulder_pitch_joint": 2.0,
  ".*_shoulder_roll_joint": 1.0,
  ".*_shoulder_yaw_joint": 0.4,
  ".*_elbow_joint": 1.0,
  ".*_wrist_roll_joint": 0.2,
  ".*_wrist_pitch_joint": 0.2,
  ".*_wrist_yaw_joint": 0.2,
}
ULC_ACTION_SCALE = 0.25


def _validate_g1_override_keys(overrides: dict[str, float], name: str) -> None:
  unknown = sorted(set(overrides) - set(G1_ACTION_SCALE))
  if unknown:
    raise KeyError(
      f"Unknown G1 {name} override keys: {unknown}. "
      "Use the upstream G1 regex keys from mjlab.asset_zoo.robots.G1_ACTION_SCALE."
    )


def _make_g1_falcon_actuators(
  base_actuators: tuple[BuiltinPositionActuatorCfg, ...],
) -> tuple[BuiltinPositionActuatorCfg, ...]:
  actuators: list[BuiltinPositionActuatorCfg] = []

  for actuator_cfg in base_actuators:
    for target_expr in actuator_cfg.target_names_expr:
      base_cfg = deepcopy(actuator_cfg)
      base_cfg.target_names_expr = (target_expr,)
      if target_expr in G1_KP_OVERRIDES:
        base_cfg.stiffness = G1_KP_OVERRIDES[target_expr]
      if target_expr in G1_KD_OVERRIDES:
        base_cfg.damping = G1_KD_OVERRIDES[target_expr]
      actuators.append(base_cfg)

  return tuple(actuators)


def unitree_g1_flat_ulc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 flat-ground ULC task configuration."""

  cfg = make_ulc_env_cfg()
  _validate_g1_override_keys(G1_KP_OVERRIDES, "Kp")
  _validate_g1_override_keys(G1_KD_OVERRIDES, "Kd")

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=_make_g1_falcon_actuators(robot_cfg.articulation.actuators),
    soft_joint_pos_limit_factor=robot_cfg.articulation.soft_joint_pos_limit_factor,
  )
  cfg.scene.entities = {"robot": robot_cfg}
  cfg.viewer.body_name = TORSO_BODY_NAME

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  undesired_contacts_cfg = ContactSensorCfg(
    name="undesired_contacts",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=(
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (feet_ground_cfg, undesired_contacts_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, UlcJointPositionActionCfg)
  joint_pos_action.scale = ULC_ACTION_SCALE
  joint_pos_action.arm_joint_names = ARM_JOINT_NAMES

  ulc_command = cfg.commands["ulc"]
  assert isinstance(ulc_command, ULCCommandCfg)
  ulc_command.arm_joint_names = ARM_JOINT_NAMES
  ulc_command.viz.z_offset = 1.15

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["payload_and_base_mass"].params["wrist_body_names"] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  cfg.events["payload_and_base_mass"].params["base_body_names"] = (
    PELVIS_BODY_NAME,
    TORSO_BODY_NAME,
  )

  cfg.rewards["upper_body"].params["asset_cfg"].joint_names = ARM_JOINT_NAMES
  for reward_name in ("torso_yaw", "torso_roll", "torso_pitch"):
    cfg.rewards[reward_name].params["torso_body_cfg"].body_names = (TORSO_BODY_NAME,)
    cfg.rewards[reward_name].params["pelvis_body_cfg"].body_names = (PELVIS_BODY_NAME,)
  cfg.rewards["center_of_mass"].params["asset_cfg"].body_names = ANKLE_BODY_NAMES
  cfg.rewards["joint_effort_limit"].params["asset_cfg"].joint_names = WAIST_JOINT_NAMES
  cfg.rewards["joint_effort_limit"].params["limits"] = WAIST_EFFORT_LIMITS
  cfg.rewards["joint_deviation"].params["minor_joint_names"] = HIP_MINOR_JOINT_NAMES
  cfg.rewards["joint_deviation"].params["major_joint_names"] = HIP_MAJOR_JOINT_NAMES
  cfg.rewards["feet_slide"].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["ankle_orientation"].params["asset_cfg"].body_names = ANKLE_BODY_NAMES

  cfg.metrics["upper_body_tracking"].params["asset_cfg"].joint_names = ARM_JOINT_NAMES
  cfg.metrics["torso_tracking"].params["torso_body_cfg"].body_names = (TORSO_BODY_NAME,)
  cfg.metrics["torso_tracking"].params["pelvis_body_cfg"].body_names = (
    PELVIS_BODY_NAME,
  )
  cfg.metrics["hip_stability"].params["minor_joint_names"] = HIP_MINOR_JOINT_NAMES
  cfg.metrics["hip_stability"].params["major_joint_names"] = HIP_MAJOR_JOINT_NAMES

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("payload_and_base_mass", None)
    cfg.curriculum = {}
    ulc_command.debug_vis = True
    ulc_command.alpha_height_init = 0.98
    ulc_command.alpha_upper_init = 0.98

    cfg.observations["critic"] = ObservationGroupCfg(
      terms=dict(cfg.observations["critic"].terms),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=6,
    )

  return cfg
