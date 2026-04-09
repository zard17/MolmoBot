"""
Batch evaluation: run multiple object combinations and generate a report.

Loads the policy once, then runs all task combinations sequentially.
Outputs a markdown report with Thor vs Objaverse success rate comparison
and a CSV file for further analysis.

Usage:
    # Generate default config
    python scripts/run_batch_eval.py --generate-config

    # Run batch evaluation
    python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json

    # Quick test (5 steps per episode)
    python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json --task_horizon_override 5
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
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
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"

RENDER_WIDTH = 640
RENDER_HEIGHT = 360
POLICY_DT_MS = 66
DESK_TOP_Z = 0.75

# Object categorization
THOR_OBJECTS = {
    "Mug_1", "Apple_1", "Egg_1", "Candle_1", "Tissue_Box_1",
    "Pencil_1", "Pen_1", "Potato_1", "Tomato_1", "Watch_1",
    "Cloth_1", "DishSponge_1", "Spatula_1", "Knife_1",
    "Cup_5", "Bowl_3", "Vase_Open_1", "Plate_1",
}

OBJECT_NAMES = {
    "Mug_1": "mug", "Apple_1": "apple", "Egg_1": "egg",
    "Candle_1": "candle", "Tissue_Box_1": "tissue box",
    "Pencil_1": "pencil", "Pen_1": "pen", "Potato_1": "potato",
    "Tomato_1": "tomato", "Watch_1": "watch", "Cloth_1": "cloth",
    "DishSponge_1": "sponge", "Spatula_1": "spatula", "Knife_1": "knife",
    "Cup_5": "cup", "Bowl_3": "bowl", "Vase_Open_1": "vase",
    "Plate_1": "plate", "bookcase": "bookcase",
}

OBJAVERSE_NAMES = {
    "45bb173c0384450487421b687bf3bf5b": "rustic shallow bowl",
    "d6fcfa410dfe402ba412cc7abe756cfd": "gray bowl",
    "cf937fff1d494219962d2031aec345aa": "pink seashell bowl",
    "07ad36c0658e4eafa91f0137e49fad58": "round gray bowl",
    "a37dbcc09514468ebcbed44cab5452f0": "rustic bowl with yellow interior",
}


def generate_default_config():
    """Generate a default batch config with recommended combinations."""
    thor_pickups = ["Mug_1", "Apple_1", "Egg_1", "Candle_1", "Tissue_Box_1"]
    thor_receptacles = ["Bowl_3", "Cup_5", "bookcase"]
    objaverse_receptacles = [
        "objaverse:45bb173c0384450487421b687bf3bf5b",
        "objaverse:d6fcfa410dfe402ba412cc7abe756cfd",
        "objaverse:cf937fff1d494219962d2031aec345aa",
    ]

    tasks = []
    # Thor pickup × Thor receptacle
    for p in thor_pickups:
        for r in thor_receptacles:
            tasks.append({"pickup": p, "receptacle": r})

    # Thor pickup × Objaverse receptacle
    for p in thor_pickups:
        for r in objaverse_receptacles:
            tasks.append({"pickup": p, "receptacle": r})

    config = {
        "task_horizon": 600,
        "repeats": 3,
        "tasks": tasks,
    }

    config_path = BENCHMARK_DIR / "batch_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    n_thor = len(thor_pickups) * len(thor_receptacles)
    n_obj = len(thor_pickups) * len(objaverse_receptacles)
    total = (n_thor + n_obj) * config["repeats"]
    print(f"Generated {config_path}")
    print(f"  Thor combinations: {n_thor}")
    print(f"  Objaverse combinations: {n_obj}")
    print(f"  Repeats per combination: {config['repeats']}")
    print(f"  Total episodes: {total}")
    print(f"  Task horizon: {config['task_horizon']} steps")


def get_object_name(uid_str: str) -> str:
    """Get human-readable name for an object uid."""
    if uid_str == "bookcase":
        return "bookcase"
    if uid_str.startswith("objaverse:"):
        hash_uid = uid_str.split(":", 1)[1]
        return OBJAVERSE_NAMES.get(hash_uid, hash_uid[:12])
    return OBJECT_NAMES.get(uid_str, uid_str)


def get_object_group(uid_str: str) -> str:
    """Categorize object as 'thor' or 'objaverse'."""
    if uid_str == "bookcase":
        return "thor"  # bookcase is our primitive, treat as thor
    if uid_str.startswith("objaverse:"):
        return "objaverse"
    return "thor"


def load_object(uid_str: str) -> Path:
    """Load an object, downloading if needed."""
    if uid_str.startswith("objaverse:"):
        hash_uid = uid_str.split(":", 1)[1]
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


def build_scene():
    """Build the base scene (furniture + robot, no objects)."""
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0],
        pos=[0, 0, 0], contype=8, conaffinity=15)

    # Desk
    desk = spec.worldbody.add_body(name="desk", pos=[0.75, 0.25, 0.0])
    desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72],
        rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"),
                        (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36],
            rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

    # Bookcase
    bc = spec.worldbody.add_body(name="bookcase", pos=[0.55, -0.55, 0.0])
    bc_c = [0.7, 0.6, 0.4, 1.0]
    bc.add_geom(name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01], rgba=bc_c, contype=8, conaffinity=15)
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc.add_geom(name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, sz], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6], rgba=bc_c, contype=8, conaffinity=15)

    # Robot
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])

    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
        fovy=71.0, resolution=[RENDER_WIDTH, RENDER_HEIGHT],
        name="robot_0/exo_camera_1")

    return spec, robot_config


def run_single_episode(spec_template, robot_config, policy, pickup_uid, receptacle_uid, prompt, task_horizon, output_dir, episode_id):
    """Run a single episode. Returns success (bool) and video path."""
    # Rebuild scene for each episode (fresh state)
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    # Rebuild furniture (can't reuse spec — MjSpec isn't copyable)
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0], pos=[0, 0, 0], contype=8, conaffinity=15)

    desk = spec.worldbody.add_body(name="desk", pos=[0.75, 0.25, 0.0])
    desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72], rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"), (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36], rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

    bc = spec.worldbody.add_body(name="bookcase", pos=[0.55, -0.55, 0.0])
    bc_c = [0.7, 0.6, 0.4, 1.0]
    bc.add_geom(name="bc_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.01, 0.8], pos=[0, -0.14, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_left", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[-0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_right", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.15, 0.8], pos=[0.29, 0, 0.8], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, 0.01], rgba=bc_c, contype=8, conaffinity=15)
    for si, sz in enumerate([0.4, 0.8, 1.2]):
        bc.add_geom(name=f"bc_shelf_{si}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.29, 0.15, 0.01], pos=[0, 0, sz], rgba=bc_c, contype=8, conaffinity=15)
    bc.add_geom(name="bc_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.15, 0.01], pos=[0, 0, 1.6], rgba=bc_c, contype=8, conaffinity=15)

    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])
    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
        fovy=71.0, resolution=[RENDER_WIDTH, RENDER_HEIGHT], name="robot_0/exo_camera_1")

    # Add pickup object on desk
    pickup_xml = load_object(pickup_uid)
    pickup_spec = mujoco.MjSpec.from_file(str(pickup_xml))
    pickup_body = pickup_spec.worldbody.bodies[0]
    if not pickup_body.first_joint():
        pickup_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    pf = spec.worldbody.add_frame(pos=[0.55, 0.25, DESK_TOP_Z + 0.04], quat=THOR_QUAT)
    pf.attach_body(pickup_body, "pickup_object/", "")

    # Add receptacle (if not bookcase)
    if receptacle_uid != "bookcase":
        recep_xml = load_object(receptacle_uid)
        recep_spec = mujoco.MjSpec.from_file(str(recep_xml))
        recep_body = recep_spec.worldbody.bodies[0]
        if not recep_body.first_joint():
            recep_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
        rf = spec.worldbody.add_frame(pos=[0.65, 0.35, DESK_TOP_Z + 0.08], quat=THOR_QUAT)
        rf.attach_body(recep_body, "place_receptacle/", "")

    model = spec.compile()
    data = mujoco.MjData(model)
    view = FrankaDroidRobotView(data, robot_config.robot_namespace)
    view.set_qpos_dict(robot_config.init_qpos)
    mujoco.mj_forward(model, data)
    for mg_id in view.move_group_ids():
        view.get_move_group(mg_id).ctrl = view.get_move_group(mg_id).noop_ctrl
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup = 0

    # Record initial pickup position for success check
    pickup_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object/" + pickup_body.name)
    initial_pickup_z = data.xpos[pickup_body_id][2] if pickup_body_id >= 0 else 0

    # Rollout
    frames = []
    policy.reset()
    for step in range(task_horizon):
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

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}/{task_horizon}", flush=True)

    # Simple success heuristic: pickup object moved significantly from start
    final_pickup_z = data.xpos[pickup_body_id][2] if pickup_body_id >= 0 else 0
    pickup_moved = abs(final_pickup_z - initial_pickup_z) > 0.05
    # TODO: more sophisticated success criteria (object in receptacle, etc.)

    # Save video
    video_path = output_dir / f"{episode_id}.mp4"
    if frames:
        ImageSequenceClip(frames, fps=round(1000 / POLICY_DT_MS)).write_videofile(
            str(video_path), codec="libx264", audio=False, logger=None)

    renderer.close()
    return pickup_moved, str(video_path)


def generate_report(results, output_dir, config):
    """Generate Markdown report and CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pickup", "receptacle", "prompt", "group", "trial", "success", "video_path"])
        for r in results:
            writer.writerow([r["pickup"], r["receptacle"], r["prompt"],
                            r["group"], r["trial"], r["success"], r["video_path"]])

    # Aggregate stats
    thor_results = [r for r in results if r["group"] == "thor"]
    objaverse_results = [r for r in results if r["group"] == "objaverse"]

    thor_success = sum(1 for r in thor_results if r["success"])
    obj_success = sum(1 for r in objaverse_results if r["success"])

    # Per-combination stats
    combo_stats = {}
    for r in results:
        key = (r["pickup"], r["receptacle"])
        if key not in combo_stats:
            combo_stats[key] = {"success": 0, "total": 0, "group": r["group"], "prompt": r["prompt"]}
        combo_stats[key]["total"] += 1
        if r["success"]:
            combo_stats[key]["success"] += 1

    # Markdown report
    md_path = output_dir / f"report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Task horizon: {config['task_horizon']} steps\n")
        f.write(f"Repeats per combination: {config['repeats']}\n\n")

        f.write("## Summary\n\n")
        f.write("| Group | Success | Total | Rate |\n")
        f.write("|-------|---------|-------|------|\n")
        if thor_results:
            f.write(f"| Thor (training) | {thor_success} | {len(thor_results)} | {thor_success/len(thor_results)*100:.1f}% |\n")
        if objaverse_results:
            f.write(f"| Objaverse (novel) | {obj_success} | {len(objaverse_results)} | {obj_success/len(objaverse_results)*100:.1f}% |\n")
        total_s = thor_success + obj_success
        total_n = len(results)
        f.write(f"| **Total** | **{total_s}** | **{total_n}** | **{total_s/total_n*100:.1f}%** |\n")

        f.write("\n## Per-Combination Results\n\n")
        f.write("| Pickup | Receptacle | Group | Success | Trials | Rate |\n")
        f.write("|--------|-----------|-------|---------|--------|------|\n")
        for (pickup, receptacle), stats in sorted(combo_stats.items()):
            rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            pickup_name = get_object_name(pickup)
            recep_name = get_object_name(receptacle)
            f.write(f"| {pickup_name} | {recep_name} | {stats['group']} | {stats['success']}/{stats['total']} | {stats['total']} | {rate:.0f}% |\n")

        f.write(f"\n## Data\n\n")
        f.write(f"- CSV: `{csv_path.name}`\n")
        f.write(f"- Videos: `{output_dir}/`\n")

    print(f"\nReport: {md_path}")
    print(f"CSV:    {csv_path}")

    # Print summary to console
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    if thor_results:
        print(f"Thor (training):    {thor_success}/{len(thor_results)} ({thor_success/len(thor_results)*100:.1f}%)")
    if objaverse_results:
        print(f"Objaverse (novel):  {obj_success}/{len(objaverse_results)} ({obj_success/len(objaverse_results)*100:.1f}%)")
    print(f"Total:              {total_s}/{total_n} ({total_s/total_n*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation with report generation")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--config", type=str, help="Path to batch config JSON")
    parser.add_argument("--generate-config", action="store_true", help="Generate default config and exit")
    parser.add_argument("--task_horizon_override", type=int, default=None, help="Override task_horizon from config")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.generate_config:
        generate_default_config()
        return

    if not args.checkpoint_path or not args.config:
        parser.error("--checkpoint_path and --config are required")

    with open(args.config) as f:
        config = json.load(f)

    task_horizon = args.task_horizon_override or config["task_horizon"]
    repeats = config["repeats"]
    tasks = config["tasks"]

    total_episodes = len(tasks) * repeats
    print(f"Batch evaluation: {len(tasks)} combinations × {repeats} repeats = {total_episodes} episodes")
    print(f"Task horizon: {task_horizon} steps")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else BENCHMARK_DIR / f"batch_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Pre-download all objects
    print("\nPre-downloading objects...")
    all_objects = set()
    for task in tasks:
        all_objects.add(task["pickup"])
        if task["receptacle"] != "bookcase":
            all_objects.add(task["receptacle"])

    for obj_uid in sorted(all_objects):
        try:
            load_object(obj_uid)
            print(f"  OK: {obj_uid}")
        except Exception as e:
            print(f"  FAIL: {obj_uid} — {e}")

    # Load policy once
    print("\nLoading policy (one-time)...")
    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    policy_config = RealRobotVLAPolicyConfig()
    policy_config.checkpoint_path = args.checkpoint_path
    policy_config.action_type = "joint_pos"
    policy_config.action_keys["arm"] = "joint_pos"

    class MockConfig:
        def __init__(self, pc):
            self.policy_config = pc

    policy = RealRobotVLAPolicy(config=MockConfig(policy_config), task_type="manipulation")
    print("Policy loaded.\n")

    # Build template scene (for robot_config)
    _, robot_config = build_scene()

    # Run all episodes
    results = []
    episode_num = 0
    for task_idx, task in enumerate(tasks):
        pickup = task["pickup"]
        receptacle = task["receptacle"]

        pickup_name = get_object_name(pickup)
        receptacle_name = get_object_name(receptacle)
        prompt = task.get("prompt") or (
            f"pick up the {pickup_name} and place it in the {receptacle_name}"
        )

        # Determine group based on receptacle (since pickups are all Thor)
        group = "objaverse" if get_object_group(receptacle) == "objaverse" else "thor"

        for trial in range(repeats):
            episode_num += 1
            episode_id = f"ep{episode_num:03d}_{pickup_name}_{receptacle_name}_t{trial}"
            print(f"[{episode_num}/{total_episodes}] {pickup_name} → {receptacle_name} (trial {trial+1}/{repeats})")

            try:
                success, video_path = run_single_episode(
                    None, robot_config, policy,
                    pickup, receptacle, prompt,
                    task_horizon, output_dir, episode_id,
                )
                status = "PASS" if success else "FAIL"
                print(f"  → {status}")
            except Exception as e:
                success = False
                video_path = ""
                print(f"  → ERROR: {e}")

            results.append({
                "pickup": pickup,
                "receptacle": receptacle,
                "prompt": prompt,
                "group": group,
                "trial": trial,
                "success": success,
                "video_path": video_path,
            })

            # Save partial results after each episode (resumable on crash)
            with open(output_dir / "results_partial.json", "w") as f:
                json.dump({"completed": episode_num, "total": total_episodes, "results": results}, f, indent=2)

    # Generate final report
    generate_report(results, output_dir, config)

    # Save final results (replace partial)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    partial = output_dir / "results_partial.json"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
