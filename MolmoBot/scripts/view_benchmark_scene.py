"""
Visualize the benchmark scene in the MuJoCo interactive viewer.

The ProcTHOR house coordinates are far from world origin (robot at ~[6.8, 9.75]),
so the default viewer camera can't see inside. This script sets the camera to
look at the robot workspace.

Usage:
    python scripts/view_benchmark_scene.py
"""

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import (
    install_scene_with_objects_and_grasps_from_path,
    install_uid,
)

ROBOT_POS = [6.8, 9.75]
ROBOT_YAW_DEG = 90.0
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)


def build_scene():
    houses = get_procthor_10k_houses(split="val")
    house_xml = houses["val"][0]["base"]
    install_scene_with_objects_and_grasps_from_path(house_xml)

    spec = mujoco.MjSpec.from_file(house_xml)

    # Add robot
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(
        robot_config,
        spec,
        robot_spec,
        prefix=robot_config.robot_namespace,
        pos=ROBOT_POS,
        quat=R.from_euler("z", ROBOT_YAW_DEG, degrees=True).as_quat(scalar_first=True),
    )

    # Add benchmark objects
    objects = {
        "pickup_object/Mug_1": {"uid": "Mug_1", "pos": [6.5, 10.1, 0.5]},
        "pickup_object/Pencil_1": {"uid": "Pencil_1", "pos": [6.6, 10.15, 0.49]},
        "place_receptacle/Cup_5": {"uid": "Cup_5", "pos": [6.9, 10.1, 0.55]},
        "place_receptacle/Shelving_Unit_206_1": {
            "uid": "Shelving_Unit_206_1",
            "pos": [7.3, 10.2, 1.1],
        },
    }

    for name, obj in objects.items():
        xml_path = install_uid(obj["uid"])
        obj_spec = mujoco.MjSpec.from_file(str(xml_path))
        body = obj_spec.worldbody.bodies[0]
        if not body.first_joint():
            body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
        frame = spec.worldbody.add_frame(pos=obj["pos"], quat=THOR_QUAT)
        prefix = name.rsplit("/", 1)[0] + "/"
        frame.attach_body(body, prefix, "")

    model = spec.compile()
    data = mujoco.MjData(model)

    # Set robot to home position
    view = FrankaDroidRobotView(data, robot_config.robot_namespace)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)

    return model, data


def main():
    print("Building scene...")
    model, data = build_scene()

    # Set default viewer camera via model keyframe
    # Camera behind the robot, looking over its shoulder toward the workspace
    # Robot is at [6.8, 9.75] facing +Y (yaw=90), so "behind" is -Y direction
    model.cam_pos0[0] = [6.8, 9.0, 1.2]  # behind and slightly above robot

    print("Launching viewer...")
    print("Controls: left-drag=rotate, right-drag=pan, scroll=zoom")
    print("Close the viewer window to exit.")
    print("Tip: on macOS use 'mjpython' instead of 'python' if launch_passive is needed.")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
