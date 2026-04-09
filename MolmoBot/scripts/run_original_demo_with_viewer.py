"""
Run the original ProcTHOR demo (from demo_policy.ipynb) with a real-time MuJoCo viewer.

This demonstrates the policy on its training-distribution scene:
ProcTHOR val house 0, with door opening and salt-shaker-to-bowl pick-and-place.

Usage:
    # Linux
    python scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

    # macOS
    mjpython scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

    # Without viewer
    python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --no-viewer

    # Specific task
    python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --task door_open
    python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --task pick_and_place
"""

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses, get_robot_path
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path, install_uid

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo_outputs" / "original_demo_viewer"

RENDER_WIDTH = 640
RENDER_HEIGHT = 360
POLICY_DT_MS = 66

ROBOT_CONFIG = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
HOUSES = get_procthor_10k_houses(split="val")
HOUSE_XML_PATH = HOUSES["val"][0]["base"]

DEFAULT_SCENE_CONFIG = {
    "robot_pos": [6.8, 9.75],
    "robot_yaw_deg": 90.0,
    "exo_pos": [0.1, 0.57, 0.66],
    "exo_quat": [-0.3633, -0.1241, 0.4263, 0.8191],
    "exo_fovy": 71.0,
}

TASK_SPECS = {
    "door_open": {
        "task_id": "door_open",
        "prompt": "open the door",
        "duration_s": 18.0,
        "scene_config": {
            "robot_pos": [8.62, 8.86],
            "robot_yaw_deg": 52.0,
            "exo_pos": [-0.05, 0.92, 0.86],
            "exo_quat": [-0.512, -0.214, 0.318, 0.771],
            "exo_fovy": 82.0,
        },
    },
    "pick_and_place": {
        "task_id": "pick_and_place",
        "prompt": "put the salt shaker in the bowl",
        "duration_s": 24.0,
        "scene_config": {},
    },
}


def build_scene(task_spec):
    scene_config = dict(DEFAULT_SCENE_CONFIG)
    scene_config.update(task_spec.get("scene_config", {}))

    install_scene_with_objects_and_grasps_from_path(HOUSE_XML_PATH)

    spec = mujoco.MjSpec.from_file(HOUSE_XML_PATH)
    robot_file_path = get_robot_path(ROBOT_CONFIG.name) / ROBOT_CONFIG.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_file_path))

    FrankaRobot.add_robot_to_scene(
        ROBOT_CONFIG, spec, robot_spec,
        prefix=ROBOT_CONFIG.robot_namespace,
        pos=scene_config["robot_pos"],
        quat=R.from_euler("z", scene_config["robot_yaw_deg"], degrees=True).as_quat(scalar_first=True),
    )

    spec.camera(ROBOT_CONFIG.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(ROBOT_CONFIG.robot_namespace + "fr3_link0").add_camera(
        pos=scene_config["exo_pos"],
        quat=scene_config["exo_quat"],
        fovy=scene_config["exo_fovy"],
        resolution=[RENDER_WIDTH, RENDER_HEIGHT],
        name="robot_0/exo_camera_1",
    )

    # Add bowl for pick-and-place task
    bowl_xml_path = install_uid("Bowl_3")
    bowl_spec = mujoco.MjSpec.from_file(str(bowl_xml_path))
    root_body = bowl_spec.worldbody.first_body()
    receptacle_frame = spec.worldbody.add_frame(
        pos=[7.1, 10.2, 1.01],
        quat=R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True),
    )
    receptacle_frame.attach_body(root_body, prefix="place_receptacle/")

    model = spec.compile()
    data = mujoco.MjData(model)
    view = FrankaDroidRobotView(data, ROBOT_CONFIG.robot_namespace)
    view.set_qpos_dict(ROBOT_CONFIG.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup = 0

    return model, data, view, renderer, scene_option, scene_config


def main():
    parser = argparse.ArgumentParser(description="Run original ProcTHOR demo with viewer")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="pick_and_place",
                        choices=["door_open", "pick_and_place"],
                        help="Task to run (default: pick_and_place)")
    parser.add_argument("--duration_s", type=float, default=None,
                        help="Override task duration in seconds")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    task_spec = TASK_SPECS[args.task]
    if args.duration_s is not None:
        task_spec = dict(task_spec, duration_s=args.duration_s)

    num_steps = round(task_spec["duration_s"] * 1000 / POLICY_DT_MS)
    print(f"Task: {task_spec['prompt']}")
    print(f"Duration: {task_spec['duration_s']}s ({num_steps} steps)")

    # Build scene
    print("Building ProcTHOR scene...")
    model, data, view, renderer, scene_option, scene_config = build_scene(task_spec)

    # Viewer
    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            # Point camera at robot workspace
            rpos = scene_config["robot_pos"]
            viewer.cam.lookat[:] = [rpos[0], rpos[1], 0.9]
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 180
            print("Viewer launched. Close window to stop early.")
        except RuntimeError:
            print("WARNING: Passive viewer not available. Use mjpython on macOS.")

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
    print(f"Running rollout ({num_steps} steps)...")
    frames = []
    for step in range(num_steps):
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

        for mg_id, ctrl in action.items():
            view.get_move_group(mg_id).ctrl = ctrl

        nstep = max(1, POLICY_DT_MS // max(1, round(model.opt.timestep * 1000)))
        mujoco.mj_step(model, data, nstep=nstep)

        if viewer:
            viewer.sync()

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    # Save video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{task_spec['task_id']}_viewer.mp4"
    if frames:
        ImageSequenceClip(frames, fps=round(1000 / POLICY_DT_MS)).write_videofile(
            str(video_path), codec="libx264", audio=False, logger=None)
        print(f"Video saved to {video_path}")

    renderer.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
