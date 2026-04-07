"""Isaac Sim scene setup helpers for RBY1 zero-shot testing.

Skeleton — fill in with Isaac Sim / Omniverse API calls.

These functions are meant to be called from an Isaac Sim extension or
standalone script that sets up the evaluation scene.
"""

from __future__ import annotations


def load_rby1_usd(
    urdf_path: str,
    stage_path: str = "/World/RBY1",
):
    """Load RBY1 URDF into Isaac Sim as a USD articulation.

    Steps:
        1. Use omni.isaac.urdf extension to convert URDF → USD
        2. Add the USD reference to the stage at stage_path
        3. Set up ArticulationController for position control
        4. Return the articulation handle

    Args:
        urdf_path: Path to the RBY1 URDF file.
            Check: MolmoBot/.venv/lib/.../molmo_spaces/assets/robots/rby1m/
            or download from the RBY1 manufacturer.
        stage_path: USD stage path for the robot prim.

    Returns:
        Articulation handle (type depends on Isaac Sim version).

    TODO:
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.isaac.urdf")

        from omni.isaac.urdf import _urdf
        urdf_interface = _urdf.acquire_urdf_interface()

        import_config = _urdf.ImportConfig()
        import_config.fix_base = False  # RBY1 has mobile base
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

        result = urdf_interface.parse_urdf(urdf_path, import_config)
        urdf_interface.import_robot(stage_path, result, import_config)

        from omni.isaac.core.articulations import Articulation
        robot = Articulation(stage_path)
        return robot
    """
    raise NotImplementedError("Fill in with Isaac Sim API calls")


def setup_cameras(
    stage_path: str = "/World/RBY1",
) -> dict[str, str]:
    """Set up camera sensors at the three RBY1 camera positions.

    Creates camera prims attached to the robot's head and wrist links,
    and sets up ROS 2 camera publishers for each.

    Args:
        stage_path: USD stage path of the RBY1 robot.

    Returns:
        Dict mapping camera name to ROS 2 topic name:
            {
                "head_camera": "/rby1/head_camera/image_raw",
                "wrist_camera_l": "/rby1/left_wrist_camera/image_raw",
                "wrist_camera_r": "/rby1/right_wrist_camera/image_raw",
            }

    TODO:
        from omni.isaac.sensor import Camera

        cameras = {}

        # Head camera — attached to head link
        head_cam = Camera(
            prim_path=f"{stage_path}/head_link/head_camera",
            resolution=(640, 480),
        )
        cameras["head_camera"] = head_cam

        # Left wrist camera
        left_cam = Camera(
            prim_path=f"{stage_path}/left_wrist_link/wrist_camera_l",
            resolution=(640, 480),
        )
        cameras["wrist_camera_l"] = left_cam

        # Right wrist camera
        right_cam = Camera(
            prim_path=f"{stage_path}/right_wrist_link/wrist_camera_r",
            resolution=(640, 480),
        )
        cameras["wrist_camera_r"] = right_cam

        # Set up ROS 2 publishers using OmniGraph or direct bridge
        # from omni.isaac.ros2_bridge import ...
        # ...
    """
    raise NotImplementedError("Fill in with Isaac Sim API calls")


def setup_joint_state_publisher(
    stage_path: str = "/World/RBY1",
    topic: str = "/rby1/joint_states",
) -> None:
    """Set up ROS 2 JointState publisher for the RBY1 robot.

    Args:
        stage_path: USD stage path of the RBY1 robot.
        topic: ROS 2 topic name.

    TODO:
        Use OmniGraph ROS 2 bridge to create a JointState publisher:
        - Create an OG graph with:
            - OnPlaybackTick node
            - IsaacReadArticulation node (prim_path=stage_path)
            - ROS2PublishJointState node (topic=topic)
        - Wire them together
    """
    raise NotImplementedError("Fill in with Isaac Sim API calls")


def setup_task_scene(
    stage_path: str = "/World",
) -> None:
    """Set up the evaluation scene: table, phone, cabinet slot.

    Args:
        stage_path: USD stage root path.

    TODO:
        - Load table USD asset
        - Load phone USD asset (graspable object)
        - Load cabinet/slot USD asset (placement target)
        - Set initial positions matching ACT task layout
        - Configure physics materials (friction, restitution)
    """
    raise NotImplementedError("Fill in with Isaac Sim API calls")
