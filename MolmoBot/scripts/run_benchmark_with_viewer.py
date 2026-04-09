"""
Run the benchmark evaluation with a real-time MuJoCo viewer.

This is a standalone script that builds the custom scene, loads the policy,
and runs the rollout with an interactive viewer window. Unlike run_eval.py
which records videos, this lets you watch the robot in real time.

Usage:
    # On Linux (X11/EGL):
    python scripts/run_benchmark_with_viewer.py --checkpoint_path <path>

    # On macOS (requires mjpython for passive viewer):
    mjpython scripts/run_benchmark_with_viewer.py --checkpoint_path <path>

    # Without viewer (just saves video):
    python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> --no-viewer

Options:
    --checkpoint_path   Path to MolmoBot-DROID checkpoint
    --task_horizon      Max steps per episode (default: 200)
    --episode           Episode index to run: 0=tissue_box, 1=pencil (default: 0)
    --no-viewer         Disable viewer, only save video
    --output_dir        Output directory for videos (default: benchmark output dir)
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import install_uid

import molmo_spaces

HOUSE_BASE_XML = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"

RENDER_WIDTH = 640
RENDER_HEIGHT = 360
POLICY_DT_MS = 66

TASK_SPECS = [
    {
        "task_id": "tissue_box_to_bookcase",
        "prompt": "pick up the tissue box and place it in the bookcase",
        "objects": [
            ("Tissue_Box_1", [0.5, 0.2, 0.79], "pickup_object/"),
        ],
    },
    {
        "task_id": "pencil_to_cup",
        "prompt": "pick up the pencil and put it in the cup",
        "objects": [
            ("Pencil_1", [0.45, 0.1, 0.77], "pickup_pencil/"),
            ("Cup_5", [0.6, 0.3, 0.83], "place_receptacle/"),
        ],
    },
]


def build_scene():
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Ground
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0], contype=8, conaffinity=15,
    )

    # Desk
    desk_body = spec.worldbody.add_body(name="desk", pos=[0.55, 0.2, 0.0])
    desk_body.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"), (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk_body.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36],
            rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

    # Bookcase
    bc_body = spec.worldbody.add_body(name="bookcase", pos=[0.5, -0.3, 0.0])
    bc_c = [0.7, 0.6, 0.4, 1.0]
    bc_body.add_geom(name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01], rgba=bc_c, contype=8, conaffinity=15)
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc_body.add_geom(name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, sz], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6], rgba=bc_c, contype=8, conaffinity=15)

    # Robot
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])

    # Wrist camera for observations
    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
        fovy=71.0, resolution=[RENDER_WIDTH, RENDER_HEIGHT],
        name="robot_0/exo_camera_1",
    )

    return spec, robot_config


def add_objects(spec, task_spec):
    for uid, pos, prefix in task_spec["objects"]:
        xml_path = install_uid(uid)
        obj_spec = mujoco.MjSpec.from_file(str(xml_path))
        body = obj_spec.worldbody.bodies[0]
        if not body.first_joint():
            body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
        frame = spec.worldbody.add_frame(pos=pos, quat=THOR_QUAT)
        frame.attach_body(body, prefix, "")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark with real-time viewer")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--task_horizon", type=int, default=200)
    parser.add_argument("--episode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    task_spec = TASK_SPECS[args.episode]
    print(f"Task: {task_spec['prompt']}")
    print(f"Steps: {args.task_horizon}")

    # Build scene
    print("Building scene...")
    spec, robot_config = build_scene()
    add_objects(spec, task_spec)

    model = spec.compile()
    data = mujoco.MjData(model)
    view = FrankaDroidRobotView(data, robot_config.robot_namespace)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    # Renderer for observations
    renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup = 0

    # Viewer
    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.lookat[:] = [0.5, 0.0, 0.8]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            print("Viewer launched. Close window to stop early.")
        except RuntimeError:
            print("WARNING: Passive viewer not available. Use mjpython on macOS.")
            print("Continuing without viewer...")

    # Load policy
    print("Loading policy...")
    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    policy_config = RealRobotVLAPolicyConfig()
    policy_config.checkpoint_path = args.checkpoint_path
    policy_config.action_type = "joint_pos"
    policy_config.action_keys["arm"] = "joint_pos"

    class MockConfig:
        def __init__(self, pc):
            self.policy_config = pc

    policy = RealRobotVLAPolicy(config=MockConfig(policy_config), task_type="manipulation")
    print("Policy loaded.")

    # Rollout
    print(f"Running rollout ({args.task_horizon} steps)...")
    frames = []
    for step in range(args.task_horizon):
        # Check viewer
        if viewer and not viewer.is_running():
            print(f"Viewer closed at step {step}")
            break

        # Render observations
        renderer.update_scene(data, camera="robot_0/exo_camera_1", scene_option=scene_option)
        exo_img = renderer.render()
        renderer.update_scene(data, camera="robot_0/gripper/wrist_camera", scene_option=scene_option)
        wrist_img = renderer.render()

        frames.append(np.hstack([exo_img, wrist_img]))

        # Get action
        jp = view.get_move_group("arm").joint_pos
        gripper = view.get_move_group("gripper").joint_pos
        obs = {
            "task": task_spec["prompt"],
            "qpos": {"arm": jp, "gripper": gripper},
            "exo_camera_1": exo_img,
            "wrist_camera": wrist_img,
        }
        action = policy.get_action(obs)

        # Apply action
        for mg_id, ctrl in action.items():
            view.get_move_group(mg_id).ctrl = ctrl

        nstep = max(1, POLICY_DT_MS // max(1, round(model.opt.timestep * 1000)))
        mujoco.mj_step(model, data, nstep=nstep)

        if viewer:
            viewer.sync()

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{args.task_horizon}")

    # Save video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{task_spec['task_id']}_viewer.mp4"
    ImageSequenceClip(frames, fps=round(1000 / POLICY_DT_MS)).write_videofile(
        str(video_path), codec="libx264", audio=False, logger=None)
    print(f"Video saved to {video_path}")

    renderer.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
