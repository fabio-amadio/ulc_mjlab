"""Unitree G1 flat ULC environment configuration."""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION_WITHOUT_SELF
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


def unitree_g1_flat_ulc_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 flat-ground ULC task configuration."""

  cfg = make_ulc_env_cfg()

  robot_cfg = get_g1_robot_cfg()
  robot_cfg.collisions = (FULL_COLLISION_WITHOUT_SELF,)
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
  joint_pos_action.scale = G1_ACTION_SCALE
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
