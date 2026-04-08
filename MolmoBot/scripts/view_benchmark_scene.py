"""
Build and visualize a minimal benchmark scene with desk, bookcase, and Franka robot.

Uses base_scene.xml instead of ProcTHOR house for a clean, reproducible layout.
Renders a preview image, then optionally launches the interactive viewer.

Usage:
    python scripts/view_benchmark_scene.py              # render preview + launch viewer
    python scripts/view_benchmark_scene.py --preview     # render preview only (no viewer)
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
import molmo_spaces
HOUSE_BASE_XML = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import install_uid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"

# Thor Y-up -> MuJoCo Z-up
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)


def attach_static(spec, uid, pos, quat=None, prefix=""):
    """Attach a Thor asset as a static body (strip any free joints)."""
    if quat is None:
        quat = THOR_QUAT
    xml_path = install_uid(uid)
    obj_spec = mujoco.MjSpec.from_file(str(xml_path))
    body = obj_spec.worldbody.bodies[0]
    # Lock free joints by setting damping very high — body effectively static
    for j in body.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            j.damping = 1e10
    frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    frame.attach_body(body, prefix, "")


def attach_dynamic(spec, uid, pos, quat=None, prefix=""):
    """Attach a Thor asset as a dynamic body (with free joint)."""
    if quat is None:
        quat = THOR_QUAT
    xml_path = install_uid(uid)
    obj_spec = mujoco.MjSpec.from_file(str(xml_path))
    body = obj_spec.worldbody.bodies[0]
    if not body.first_joint():
        body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    frame.attach_body(body, prefix, "")


def build_scene():
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Ground plane
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0],
        contype=8,
        conaffinity=15,
    )

    # --- Furniture (static) ---
    # Desk to the left of robot (+Y side), pushed forward and away to clear pedestal
    # Lower z from 0.38 to 0.25 so legs sit closer to ground
    attach_static(spec, "RoboTHOR_desk_lisabo", pos=[0.8, 0.55, 0.25], prefix="desk/")

    # Bookcase to the right of robot (-Y side), rotated 180° around Z
    # so open shelves face the robot (+Y direction).
    # x-90 converts Y-up to Z-up, then z-180 flips the front to face +Y.
    shelf_quat = (R.from_euler("z", 180, degrees=True) * R.from_euler("x", 90, degrees=True)).as_quat(scalar_first=True)
    attach_static(spec, "Shelving_Unit_206_1", pos=[0.55, -0.5, 0.35], quat=shelf_quat, prefix="bookcase/")

    # --- Robot ---
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))

    # Robot at origin, facing +X (toward desk)
    FrankaRobot.add_robot_to_scene(
        robot_config,
        spec,
        robot_spec,
        prefix=robot_config.robot_namespace,
        pos=[0, 0],
        quat=[1, 0, 0, 0],  # facing +X
    )

    # --- Manipulable objects (dynamic) ---
    desk_surface_z = 0.63  # lowered to match new desk z

    # Tissue box on the desk (task 1: pick and place in bookcase)
    attach_dynamic(spec, "Tissue_Box_1", pos=[0.7, 0.5, desk_surface_z + 0.08], prefix="pickup_object/")

    # Pencil on the desk (task 2: pick and place in cup)
    attach_dynamic(spec, "Pencil_1", pos=[0.7, 0.35, desk_surface_z + 0.02], prefix="pickup_pencil/")

    # Cup on the desk (receptacle for pencil)
    attach_dynamic(spec, "Cup_5", pos=[0.6, 0.6, desk_surface_z + 0.08], prefix="place_receptacle/")

    # Compile
    model = spec.compile()
    data = mujoco.MjData(model)

    # Set robot to home position
    view = FrankaDroidRobotView(data, robot_config.robot_namespace)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)

    return model, data


def render_preview(model, data, save_dir=None):
    """Render multiple preview angles."""
    w, h = 1280, 720
    renderer = mujoco.Renderer(model, h, w)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup = 0

    views = {
        "behind": {"lookat": [0.5, 0.0, 0.8], "dist": 2.5, "elev": -20, "azim": 180},
        "side": {"lookat": [0.5, -0.3, 0.8], "dist": 3.0, "elev": -15, "azim": 90},
        "top": {"lookat": [0.5, -0.3, 0.5], "dist": 3.5, "elev": -75, "azim": 180},
    }

    images = {}
    for name, v in views.items():
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = v["lookat"]
        camera.distance = v["dist"]
        camera.elevation = v["elev"]
        camera.azimuth = v["azim"]
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        img = renderer.render()
        pil_img = Image.fromarray(img)
        images[name] = pil_img
        if save_dir:
            path = Path(save_dir) / f"scene_preview_{name}.png"
            pil_img.save(path)
            print(f"Preview saved to {path}")

    renderer.close()
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Render preview only, no viewer")
    args = parser.parse_args()

    print("Building scene...")
    model, data = build_scene()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_preview(model, data, save_dir=OUTPUT_DIR)

    if args.preview:
        return

    print("Launching viewer...")
    print("Controls: left-drag=rotate, right-drag=pan, scroll=zoom")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
