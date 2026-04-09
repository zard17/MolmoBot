"""
Run the benchmark scene with interchangeable objects and a real-time MuJoCo viewer.

Supports swapping pickup and receptacle objects via CLI args to test
generalization across different object combinations.

Usage:
    # Default (tissue box → bookcase)
    python scripts/run_benchmark_with_viewer.py --checkpoint_path <path>

    # Custom objects
    python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
        --pickup Candle_1 --receptacle Vase_Open_1 \
        --prompt "put the candle in the vase"

    # List available objects
    python scripts/run_benchmark_with_viewer.py --list

    # Objaverse object (download on first use)
    python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
        --pickup Candle_1 --receptacle objaverse:45bb173c0384450487421b687bf3bf5b \
        --prompt "put the candle in the bowl"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_path
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

# Desk surface z (desk at [0.75, 0.25, 0], top at 0.72 + 0.015 half-thickness)
DESK_TOP_Z = 0.75

# Common Thor objects that are rigid and graspable
AVAILABLE_OBJECTS = {
    # Containers / receptacles
    "Cup_5": "cup",
    "Bowl_3": "bowl",
    "Vase_Open_1": "vase",
    "Plate_1": "plate",
    # Small graspable objects
    "Tissue_Box_1": "tissue box",
    "Pencil_1": "pencil",
    "Pen_1": "pen",
    "Candle_1": "candle",
    "Cloth_1": "cloth",
    "Watch_1": "watch",
    "Egg_1": "egg",
    "Apple_1": "apple",
    "Potato_1": "potato",
    "Tomato_1": "tomato",
    "Mug_1": "mug",
    "Knife_1": "knife",
    "Spatula_1": "spatula",
    "DishSponge_1": "sponge",
}


def load_object(uid_str: str) -> Path:
    """Load an object by Thor uid or objaverse:hash format."""
    if uid_str.startswith("objaverse:"):
        hash_uid = uid_str.split(":", 1)[1]
        # Download if needed
        from molmo_spaces.molmo_spaces_constants import get_resource_manager
        rm = get_resource_manager()
        try:
            archives = rm.index_lookup("objects", "objaverse", hash_uid)
            rm.install_packages("objects", {"objaverse": archives})
        except Exception:
            pass
        return install_uid(hash_uid)
    else:
        return install_uid(uid_str)


# Curated Objaverse objects (novel, not in Thor training set)
# These are downloaded on first use via the resource manager.
OBJAVERSE_OBJECTS = {
    "45bb173c0384450487421b687bf3bf5b": "rustic shallow bowl [receptacle]",
    "d6fcfa410dfe402ba412cc7abe756cfd": "gray bowl [receptacle]",
    "cf937fff1d494219962d2031aec345aa": "pink seashell bowl [receptacle]",
    "07ad36c0658e4eafa91f0137e49fad58": "round gray bowl [receptacle]",
    "a37dbcc09514468ebcbed44cab5452f0": "rustic bowl with yellow interior [receptacle]",
}


def list_objects():
    """Print available objects."""
    print("=== Thor objects (training distribution) ===")
    print(f"  {'UID':25s} {'Name':15s}")
    print(f"  {'-'*25} {'-'*15}")
    for uid, name in sorted(AVAILABLE_OBJECTS.items()):
        print(f"  {uid:25s} {name:15s}")
    print()
    print("=== Objaverse objects (novel, use objaverse:<hash>) ===")
    print(f"  {'Hash':40s} {'Description'}")
    print(f"  {'-'*40} {'-'*40}")
    for hash_uid, desc in OBJAVERSE_OBJECTS.items():
        print(f"  {hash_uid:40s} {desc}")
    print()
    print("You can also use any Objaverse hash: objaverse:<hash>")
    print("Objects are downloaded on first use.")


def build_scene():
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Ground
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0], contype=8, conaffinity=15,
    )

    # Desk
    desk_body = spec.worldbody.add_body(name="desk", pos=[0.75, 0.25, 0.0])
    desk_body.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"),
                        (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk_body.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36],
            rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

    # Bookcase
    bc_body = spec.worldbody.add_body(name="bookcase", pos=[0.55, -0.55, 0.0])
    bc_c = [0.7, 0.6, 0.4, 1.0]
    bc_body.add_geom(name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01], rgba=bc_c, contype=8, conaffinity=15)
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc_body.add_geom(name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.29, 0.15, 0.01], pos=[0, 0, sz], rgba=bc_c, contype=8, conaffinity=15)
    bc_body.add_geom(name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6], rgba=bc_c, contype=8, conaffinity=15)

    # Robot
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])

    # Cameras
    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
        fovy=71.0, resolution=[RENDER_WIDTH, RENDER_HEIGHT],
        name="robot_0/exo_camera_1",
    )

    return spec, robot_config


def add_object_to_scene(spec, uid_str, pos, prefix):
    """Add a single object to the scene."""
    xml_path = load_object(uid_str)
    obj_spec = mujoco.MjSpec.from_file(str(xml_path))
    body = obj_spec.worldbody.bodies[0]
    if not body.first_joint():
        body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    frame = spec.worldbody.add_frame(pos=pos, quat=THOR_QUAT)
    frame.attach_body(body, prefix, "")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark with interchangeable objects and viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default task (tissue box → bookcase)
  python scripts/run_benchmark_with_viewer.py --checkpoint_path <path>

  # Custom pickup + receptacle on desk
  python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \\
      --pickup Candle_1 --receptacle Bowl_3 \\
      --prompt "put the candle in the bowl"

  # Objaverse novel object
  python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \\
      --pickup Egg_1 --receptacle objaverse:45bb173c0384450487421b687bf3bf5b \\
      --prompt "put the egg in the bowl"

  # List available objects
  python scripts/run_benchmark_with_viewer.py --list
        """,
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--task_horizon", type=int, default=200)
    parser.add_argument("--pickup", type=str, default="Tissue_Box_1",
                        help="Pickup object uid (default: Tissue_Box_1)")
    parser.add_argument("--receptacle", type=str, default="bookcase",
                        help="Receptacle: 'bookcase' (shelf) or object uid (default: bookcase)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Task prompt (auto-generated if not specified)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--list", action="store_true", help="List available objects and exit")
    args = parser.parse_args()

    if args.list:
        list_objects()
        return

    if not args.checkpoint_path:
        parser.error("--checkpoint_path is required (unless using --list)")

    # Determine object names for prompt
    pickup_name = AVAILABLE_OBJECTS.get(args.pickup, args.pickup.split(":")[-1])
    if args.receptacle == "bookcase":
        receptacle_name = "bookcase"
    else:
        receptacle_name = AVAILABLE_OBJECTS.get(args.receptacle, args.receptacle.split(":")[-1])

    # Auto-generate prompt if not specified
    if args.prompt:
        prompt = args.prompt
    elif args.receptacle == "bookcase":
        prompt = f"pick up the {pickup_name} and place it in the bookcase"
    else:
        prompt = f"pick up the {pickup_name} and place it in the {receptacle_name}"

    task_id = f"{pickup_name}_to_{receptacle_name}".replace(" ", "_")
    print(f"Task: {prompt}")
    print(f"Pickup: {args.pickup} ({pickup_name})")
    print(f"Receptacle: {args.receptacle} ({receptacle_name})")
    print(f"Steps: {args.task_horizon}")

    # Build scene
    print("Building scene...")
    spec, robot_config = build_scene()

    # Add pickup object on desk
    print(f"  Loading pickup: {args.pickup}")
    add_object_to_scene(spec, args.pickup,
                        pos=[0.55, 0.25, DESK_TOP_Z + 0.04],
                        prefix="pickup_object/")

    # Add receptacle (either on desk or use bookcase)
    if args.receptacle != "bookcase":
        print(f"  Loading receptacle: {args.receptacle}")
        add_object_to_scene(spec, args.receptacle,
                            pos=[0.65, 0.35, DESK_TOP_Z + 0.08],
                            prefix="place_receptacle/")
    else:
        print("  Using bookcase as receptacle")

    model = spec.compile()
    data = mujoco.MjData(model)
    view = FrankaDroidRobotView(data, robot_config.robot_namespace)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        mg = view.get_move_group(mg_id)
        mg.ctrl = mg.noop_ctrl
    mujoco.mj_forward(model, data)

    # Renderer
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
        if viewer and not viewer.is_running():
            print(f"Viewer closed at step {step}")
            break

        renderer.update_scene(data, camera="robot_0/exo_camera_1", scene_option=scene_option)
        exo_img = renderer.render()
        renderer.update_scene(data, camera="robot_0/gripper/wrist_camera", scene_option=scene_option)
        wrist_img = renderer.render()

        frames.append(np.hstack([exo_img, wrist_img]))

        jp = view.get_move_group("arm").joint_pos
        gripper = view.get_move_group("gripper").joint_pos
        obs = {
            "task": prompt,
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
            print(f"  Step {step + 1}/{args.task_horizon}")

    # Save video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{task_id}_viewer.mp4"
    if frames:
        ImageSequenceClip(frames, fps=round(1000 / POLICY_DT_MS)).write_videofile(
            str(video_path), codec="libx264", audio=False, logger=None)
        print(f"Video saved to {video_path}")

    renderer.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
