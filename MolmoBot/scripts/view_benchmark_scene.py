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

    # --- Furniture (MuJoCo primitives for clean collision) ---
    # Thor furniture assets have oversized collision boxes that cause penetration.
    # Build desk and bookcase from MuJoCo box geoms instead.

    # Desk: 1.0m wide, 0.5m deep, 0.72m tall
    # Robot base is 0.5x0.5m → extends to x=0.25. Desk nearest leg at x-0.45,
    # so desk center x must be >= 0.25 + 0.45 + margin = 0.75
    desk_body = spec.worldbody.add_body(name="desk", pos=[0.75, 0.25, 0.0])
    # Table top
    desk_body.add_geom(
        name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15,
    )
    # Four legs
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"), (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk_body.add_geom(
            name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36],
            rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15,
        )

    # Bookcase: 0.6m wide, 0.3m deep, open front facing +Y
    # Back at y=-0.14, sides at x=±0.29. Keep clear of robot base (y=-0.25).
    bc_body = spec.worldbody.add_body(name="bookcase", pos=[0.55, -0.55, 0.0])
    bc_color = [0.7, 0.6, 0.4, 1.0]
    # Back panel
    bc_body.add_geom(
        name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8],
        rgba=bc_color, contype=8, conaffinity=15,
    )
    # Left side
    bc_body.add_geom(
        name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8],
        rgba=bc_color, contype=8, conaffinity=15,
    )
    # Right side
    bc_body.add_geom(
        name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8],
        rgba=bc_color, contype=8, conaffinity=15,
    )
    # Bottom shelf
    bc_body.add_geom(
        name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01],
        rgba=bc_color, contype=8, conaffinity=15,
    )
    # Shelves at 0.4, 0.8, 1.2m
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc_body.add_geom(
            name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.29, 0.15, 0.01], pos=[0, 0, sz],
            rgba=bc_color, contype=8, conaffinity=15,
        )
    # Top
    bc_body.add_geom(
        name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6],
        rgba=bc_color, contype=8, conaffinity=15,
    )

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
    # Desk at [0.55, 0.2], top geom z=0.72, half-thickness=0.015 → surface at 0.735
    desk_top_z = 0.75  # surface + margin

    # Objects on desk: center [0.75,0.25], half-size [0.5,0.25]
    # → x range [0.25..1.25], y range [0.0..0.5]

    # Tissue box on desk (near robot)
    attach_dynamic(spec, "Tissue_Box_1", pos=[0.55, 0.25, desk_top_z + 0.04], prefix="pickup_object/")

    # Pencil on desk
    attach_dynamic(spec, "Pencil_1", pos=[0.5, 0.15, desk_top_z + 0.02], prefix="pickup_pencil/")

    # Cup on desk
    attach_dynamic(spec, "Cup_5", pos=[0.65, 0.35, desk_top_z + 0.08], prefix="place_receptacle/")

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
