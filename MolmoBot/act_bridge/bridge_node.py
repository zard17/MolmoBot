"""ROS 2 bridge node: Isaac Sim ↔ MolmoBot policy server.

Skeleton — fill in ROS 2 API calls when integrating in the ACT repo.

Data flow:
    Isaac Sim ──[ROS 2 topics]──> BridgeNode ──[WebSocket]──> MolmoBot Server
    Isaac Sim <──[ROS 2 topics]── BridgeNode <──[WebSocket]── MolmoBot Server

Dependencies (NOT in MolmoBot's requirements — install in ACT environment):
    - rclpy
    - sensor_msgs, std_msgs, trajectory_msgs
    - cv_bridge
    - olmo.eval.websocket_client (from MolmoBot)
"""

from __future__ import annotations

import logging

import numpy as np

# TODO: Uncomment when running in a ROS 2 environment
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, JointState
# from std_msgs.msg import String
# from cv_bridge import CvBridge

from olmo.eval.websocket_client import MolmoBotClient, build_rby1_obs

from .config import ACTBridgeConfig

logger = logging.getLogger(__name__)


class MolmoBotBridgeNode:  # TODO: inherit from Node
    """ROS 2 node bridging Isaac Sim to MolmoBot policy server.

    Subscribes:
        - 3 camera image topics (sensor_msgs/Image)
        - 1 joint state topic (sensor_msgs/JointState)

    Publishes:
        - 1 joint command topic

    Internal:
        - MolmoBotClient for WebSocket communication
        - Timer callback at control_rate_hz
    """

    def __init__(self, config: ACTBridgeConfig | None = None):
        # TODO: super().__init__("molmobot_bridge")
        self.config = config or ACTBridgeConfig()

        # -- State buffers (populated by subscriber callbacks) --
        self._camera_frames: dict[str, np.ndarray | None] = {
            name: None for name in self.config.camera_topics.values()
        }
        self._joint_positions: dict[str, np.ndarray | None] = {
            group: None for group in self.config.joint_groups
        }
        self._task: str = self.config.default_task

        # -- WebSocket client --
        self._client = MolmoBotClient(
            self.config.molmobot_host, self.config.molmobot_port
        )

        # TODO: cv_bridge for Image → numpy conversion
        # self._cv_bridge = CvBridge()

        self._setup_ros()

    def _setup_ros(self) -> None:
        """Create ROS 2 subscribers, publisher, and timer."""
        # TODO: Camera subscribers
        # for topic, cam_name in self.config.camera_topics.items():
        #     self.create_subscription(
        #         Image, topic,
        #         lambda msg, cn=cam_name: self._camera_callback(msg, cn),
        #         10,
        #     )

        # TODO: Joint state subscriber
        # self.create_subscription(
        #     JointState, self.config.joint_state_topic,
        #     self._joint_state_callback, 10,
        # )

        # TODO: Joint command publisher
        # self._joint_cmd_pub = self.create_publisher(
        #     JointState, self.config.joint_command_topic, 10,
        # )

        # TODO: Timer for control loop
        # period = 1.0 / self.config.control_rate_hz
        # self.create_timer(period, self._timer_callback)

        logger.info(
            f"Bridge configured: {self.config.molmobot_host}:{self.config.molmobot_port} "
            f"@ {self.config.control_rate_hz}Hz"
        )

    def _camera_callback(self, msg, camera_name: str) -> None:
        """Store latest camera frame as numpy array.

        Args:
            msg: sensor_msgs/Image message.
            camera_name: MolmoBot camera key (e.g. "wrist_camera_r").
        """
        # TODO: Convert ROS Image to numpy
        # frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # self._camera_frames[camera_name] = frame
        pass

    def _joint_state_callback(self, msg) -> None:
        """Store latest joint positions, split into MolmoBot groups.

        Args:
            msg: sensor_msgs/JointState message.
        """
        # TODO: Build a name→position lookup from msg.name and msg.position
        # joint_lookup = dict(zip(msg.name, msg.position))
        #
        # for group_name, joint_names in self.config.joint_groups.items():
        #     values = [joint_lookup.get(jn, 0.0) for jn in joint_names]
        #     self._joint_positions[group_name] = np.array(values, dtype=np.float32)
        pass

    def _timer_callback(self) -> None:
        """Main control loop: gather obs → call policy → publish commands."""
        # Check all data is fresh
        if not self._all_data_ready():
            return

        # Build observation
        obs = build_rby1_obs(
            head_camera=self._camera_frames["head_camera"],
            wrist_camera_l=self._camera_frames["wrist_camera_l"],
            wrist_camera_r=self._camera_frames["wrist_camera_r"],
            base=self._joint_positions["base"],
            left_arm=self._joint_positions["left_arm"],
            left_gripper=self._joint_positions["left_gripper"],
            right_arm=self._joint_positions["right_arm"],
            right_gripper=self._joint_positions["right_gripper"],
            torso=self._joint_positions["torso"],
            task=self._task,
        )

        # Get action from MolmoBot server
        action = self._client.get_action(obs)

        # Convert to joint commands and publish
        joint_cmd = self._action_to_joint_command(action)
        self._publish_joint_command(joint_cmd)

    def _all_data_ready(self) -> bool:
        """Check that all camera frames and joint positions have been received."""
        if any(v is None for v in self._camera_frames.values()):
            return False
        if any(v is None for v in self._joint_positions.values()):
            return False
        return True

    def _action_to_joint_command(self, action: dict) -> dict[str, np.ndarray]:
        """Convert MolmoBot action dict to absolute joint positions.

        Delta groups (base, left_arm, right_arm):
            target = current_position + delta
        Absolute groups (left_gripper, right_gripper, torso):
            target = action_value (with gripper range remapping)
        """
        joint_cmd = {}

        # Delta groups: add to current position
        for group in self.config.delta_groups:
            current = self._joint_positions[group]
            joint_cmd[group] = current + action[group]

        # Absolute groups
        for group in self.config.absolute_groups:
            if "gripper" in group:
                joint_cmd[group] = self._remap_gripper(action[group])
            else:
                joint_cmd[group] = action[group]

        return joint_cmd

    def _remap_gripper(self, molmobot_value: np.ndarray) -> np.ndarray:
        """Map MolmoBot gripper value (±100) to Isaac Sim range (meters)."""
        cfg = self.config
        # MolmoBot: -100 = closed, +100 = open
        # Normalize to [0, 1]
        t = (molmobot_value - cfg.gripper_molmobot_close) / (
            cfg.gripper_molmobot_open - cfg.gripper_molmobot_close
        )
        t = np.clip(t, 0.0, 1.0)
        # Scale to sim range
        return cfg.gripper_sim_close + t * (cfg.gripper_sim_open - cfg.gripper_sim_close)

    def _publish_joint_command(self, joint_cmd: dict[str, np.ndarray]) -> None:
        """Publish joint command to ROS 2 topic.

        Args:
            joint_cmd: Dict of group_name → target joint positions.
        """
        # TODO: Build JointState or JointTrajectory message
        # msg = JointState()
        # for group_name, positions in joint_cmd.items():
        #     joint_names = self.config.joint_groups[group_name]
        #     msg.name.extend(joint_names)
        #     msg.position.extend(positions.tolist())
        # self._joint_cmd_pub.publish(msg)
        pass

    def connect_to_server(self) -> dict:
        """Connect to MolmoBot server. Call before spinning the node."""
        return self._client.connect()

    def disconnect(self) -> None:
        """Disconnect from MolmoBot server."""
        self._client.close()


def main():
    """Entry point for the ROS 2 bridge node."""
    # TODO: Uncomment when running in a ROS 2 environment
    # rclpy.init()
    # config = ACTBridgeConfig()
    # node = MolmoBotBridgeNode(config)
    # node.connect_to_server()
    # try:
    #     rclpy.spin(node)
    # finally:
    #     node.disconnect()
    #     node.destroy_node()
    #     rclpy.shutdown()
    print("Bridge node skeleton — run in a ROS 2 environment.")


if __name__ == "__main__":
    main()
