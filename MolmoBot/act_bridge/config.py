"""Bridge configuration: all mapping constants between ACT (Isaac Sim) and MolmoBot.

This is a reference skeleton. Adapt into your ROS 2 package when integrating.
Joint names are placeholders — confirm after loading RBY1 URDF into Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ACTBridgeConfig:
    """Configuration for the ROS 2 ↔ MolmoBot WebSocket bridge."""

    # -- MolmoBot server connection --
    molmobot_host: str = "localhost"
    molmobot_port: int = 8000

    # -- Control rate --
    # MolmoBot RBY1 policy runs at ~100ms per inference step.
    # Target 10 Hz to stay within the policy's designed cadence.
    control_rate_hz: float = 10.0

    # -- Camera topic → MolmoBot camera name mapping --
    # Keys: ROS 2 topic names (sensor_msgs/Image)
    # Values: MolmoBot observation dict keys
    camera_topics: dict[str, str] = field(default_factory=lambda: {
        "/rby1/right_wrist_camera/image_raw": "wrist_camera_r",
        "/rby1/left_wrist_camera/image_raw": "wrist_camera_l",
        "/rby1/head_camera/image_raw": "head_camera",
    })

    # -- Joint state topic --
    joint_state_topic: str = "/rby1/joint_states"

    # -- Joint command topic --
    joint_command_topic: str = "/rby1/joint_commands"

    # -- Isaac Sim joint name → MolmoBot qpos group mapping --
    # TODO: Confirm joint names after loading RBY1 URDF into Isaac Sim.
    # These are placeholder names based on typical RBY1 URDF conventions.
    joint_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "base": [
            "base_x_joint",
            "base_y_joint",
            "base_rz_joint",
        ],
        "left_arm": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
        ],
        "left_gripper": [
            "left_gripper_joint",
        ],
        "right_arm": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ],
        "right_gripper": [
            "right_gripper_joint",
        ],
        "torso": [
            "torso_joint_0",
            "torso_joint_1",
            "torso_joint_2",
            "torso_joint_3",
            "torso_joint_4",
            "torso_joint_5",
        ],
    })

    # -- Action interpretation --
    # From MolmoBotRBY1MultitaskPolicyConfig.action_keys:
    #   joint_pos_rel = delta (add to current position)
    #   joint_pos     = absolute (use directly)
    delta_groups: list[str] = field(default_factory=lambda: [
        "base", "left_arm", "right_arm",
    ])
    absolute_groups: list[str] = field(default_factory=lambda: [
        "left_gripper", "right_gripper", "torso",
    ])

    # -- Gripper range mapping --
    # MolmoBot outputs ±100 for gripper (thresholded binary).
    # Isaac Sim uses physical units (meters of jaw opening).
    gripper_molmobot_open: float = 100.0    # MolmoBot value for "open"
    gripper_molmobot_close: float = -100.0  # MolmoBot value for "close"
    gripper_sim_open: float = 0.04          # meters, TODO: confirm for RBY1 in Isaac Sim
    gripper_sim_close: float = 0.0          # meters

    # -- Task description --
    default_task: str = "pick the phone and place it in the slot"
