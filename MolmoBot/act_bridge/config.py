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
    # MuJoCo joint names from molmo_spaces/robots/robot_views/rby1_view.py.
    # Isaac Sim names may differ after URDF/USD import — confirm and update.
    joint_groups: dict[str, list[str]] = field(default_factory=lambda: {
        "base": [
            "base_x",       # holonomic base x
            "base_y",       # holonomic base y
            "base_theta",   # holonomic base yaw
        ],
        "left_arm": [
            "left_arm_0",
            "left_arm_1",
            "left_arm_2",
            "left_arm_3",
            "left_arm_4",
            "left_arm_5",
            "left_arm_6",
        ],
        "left_gripper": [
            "gripper_finger_l1",  # 2 coupled fingers, use finger 1 as control
        ],
        "right_arm": [
            "right_arm_0",
            "right_arm_1",
            "right_arm_2",
            "right_arm_3",
            "right_arm_4",
            "right_arm_5",
            "right_arm_6",
        ],
        "right_gripper": [
            "gripper_finger_r1",  # 2 coupled fingers, use finger 1 as control
        ],
        "torso": [
            "torso_0",
            "torso_1",
            "torso_2",
            "torso_3",
            "torso_4",
            "torso_5",
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
