"""
Batch evaluation: run multiple object combinations and generate a report.

Supports two scene modes:
  - "procthor": Original ProcTHOR val house 0 (training distribution)
  - "custom": Custom scene with primitive desk + bookcase

Generates a markdown report comparing success rates across:
  - Scene type (ProcTHOR vs custom)
  - Object type (Thor training objects vs Objaverse novel objects)

Usage:
    # Generate default config (4 groups: scene × object type)
    python scripts/run_batch_eval.py --generate-config

    # Run batch evaluation
    python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json

    # Quick test (10 steps per episode)
    python scripts/run_batch_eval.py --checkpoint_path <path> --config batch_config.json --task_horizon_override 10
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
import numpy as np
from scipy.spatial.transform import Rotation as R
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_robot_path, get_procthor_10k_houses
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView
from molmo_spaces.utils.lazy_loading_utils import install_uid, install_scene_with_objects_and_grasps_from_path

import molmo_spaces

HOUSE_BASE_XML = Path(molmo_spaces.__file__).parent / "resources" / "base_scene.xml"
THOR_QUAT = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "franka_book_pencil_pick_place"

RENDER_WIDTH = 640
RENDER_HEIGHT = 360
POLICY_DT_MS = 66

# Custom scene desk surface
CUSTOM_DESK_TOP_Z = 0.75

# ProcTHOR scene: objects placed on countertop near robot at [6.8, 9.75]
PROCTHOR_PICKUP_POS = [6.5, 10.1, 0.96]
PROCTHOR_RECEPTACLE_POS = [7.1, 10.2, 1.01]
PROCTHOR_ROBOT_POS = [6.8, 9.75]
PROCTHOR_ROBOT_YAW = 90.0
PROCTHOR_EXO_POS = [0.1, 0.57, 0.66]
PROCTHOR_EXO_QUAT = [-0.3633, -0.1241, 0.4263, 0.8191]
PROCTHOR_EXO_FOVY = 71.0

OBJECT_NAMES = {
    "Mug_1": "mug", "Apple_1": "apple", "Egg_1": "egg",
    "Candle_1": "candle", "Tissue_Box_1": "tissue box",
    "Pencil_1": "pencil", "Pen_1": "pen", "Potato_1": "potato",
    "Tomato_1": "tomato", "Watch_1": "watch", "Cloth_1": "cloth",
    "DishSponge_1": "sponge", "Spatula_1": "spatula", "Knife_1": "knife",
    "Cup_5": "cup", "Bowl_3": "bowl", "Vase_Open_1": "vase",
    "Plate_1": "plate", "bookcase": "bookcase",
    "SaltShaker_1": "salt shaker",
}

OBJAVERSE_NAMES = {
    "45bb173c0384450487421b687bf3bf5b": "rustic shallow bowl",
    "d6fcfa410dfe402ba412cc7abe756cfd": "gray bowl",
    "cf937fff1d494219962d2031aec345aa": "pink seashell bowl",
}


def generate_default_config():
    """Generate default config with 4 groups for comparison."""
    thor_pickups = ["SaltShaker_1", "Mug_1", "Egg_1", "Candle_1", "Tomato_1"]
    thor_receptacles = ["Bowl_3"]
    objaverse_receptacles = [
        "objaverse:45bb173c0384450487421b687bf3bf5b",
        "objaverse:d6fcfa410dfe402ba412cc7abe756cfd",
    ]

    tasks = []

    # Group A: ProcTHOR scene + Thor objects (baseline — closest to training)
    for p in thor_pickups:
        for r in thor_receptacles:
            tasks.append({"scene": "procthor", "pickup": p, "receptacle": r})

    # Group B: ProcTHOR scene + Objaverse receptacles (object generalization)
    for p in thor_pickups:
        for r in objaverse_receptacles:
            tasks.append({"scene": "procthor", "pickup": p, "receptacle": r})

    # Group C: Custom scene + Thor objects (scene generalization)
    for p in thor_pickups:
        for r in thor_receptacles:
            tasks.append({"scene": "custom", "pickup": p, "receptacle": r})

    # Group D: Custom scene + Objaverse receptacles (scene + object generalization)
    for p in thor_pickups:
        for r in objaverse_receptacles:
            tasks.append({"scene": "custom", "pickup": p, "receptacle": r})

    config = {
        "task_horizon": 600,
        "repeats": 3,
        "tasks": tasks,
    }

    config_path = BENCHMARK_DIR / "batch_config.json"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Count by group
    groups = {}
    for t in tasks:
        g = get_group_name(t["scene"], t.get("receptacle", ""))
        groups[g] = groups.get(g, 0) + 1

    total = len(tasks) * config["repeats"]
    print(f"Generated {config_path}")
    print(f"  Groups:")
    for g, n in groups.items():
        print(f"    {g}: {n} combinations × {config['repeats']} repeats = {n * config['repeats']} episodes")
    print(f"  Total episodes: {total}")
    print(f"  Task horizon: {config['task_horizon']} steps")


def get_object_name(uid_str: str) -> str:
    if uid_str == "bookcase":
        return "bookcase"
    if uid_str.startswith("objaverse:"):
        hash_uid = uid_str.split(":", 1)[1]
        return OBJAVERSE_NAMES.get(hash_uid, hash_uid[:12])
    return OBJECT_NAMES.get(uid_str, uid_str)


def get_object_type(uid_str: str) -> str:
    if uid_str.startswith("objaverse:"):
        return "objaverse"
    return "thor"


def get_group_name(scene: str, receptacle: str) -> str:
    obj_type = get_object_type(receptacle)
    if scene == "procthor" and obj_type == "thor":
        return "A: ProcTHOR+Thor (baseline)"
    elif scene == "procthor" and obj_type == "objaverse":
        return "B: ProcTHOR+Objaverse (obj gen)"
    elif scene == "custom" and obj_type == "thor":
        return "C: Custom+Thor (scene gen)"
    else:
        return "D: Custom+Objaverse (both gen)"


def load_object(uid_str: str) -> Path:
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


def build_procthor_scene(robot_config, pickup_uid, receptacle_uid):
    """Build ProcTHOR val house 0 scene with swapped objects."""
    houses = get_procthor_10k_houses(split="val")
    house_xml = houses["val"][0]["base"]
    install_scene_with_objects_and_grasps_from_path(house_xml)

    spec = mujoco.MjSpec.from_file(house_xml)
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))

    FrankaRobot.add_robot_to_scene(
        robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace,
        pos=PROCTHOR_ROBOT_POS,
        quat=R.from_euler("z", PROCTHOR_ROBOT_YAW, degrees=True).as_quat(scalar_first=True),
    )

    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=PROCTHOR_EXO_POS, quat=PROCTHOR_EXO_QUAT, fovy=PROCTHOR_EXO_FOVY,
        resolution=[RENDER_WIDTH, RENDER_HEIGHT], name="robot_0/exo_camera_1",
    )

    # Add pickup object on countertop
    pickup_xml = load_object(pickup_uid)
    pickup_spec = mujoco.MjSpec.from_file(str(pickup_xml))
    pickup_body = pickup_spec.worldbody.bodies[0]
    if not pickup_body.first_joint():
        pickup_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    pf = spec.worldbody.add_frame(pos=PROCTHOR_PICKUP_POS, quat=THOR_QUAT)
    pf.attach_body(pickup_body, "pickup_object/", "")

    # Add receptacle
    recep_xml = load_object(receptacle_uid)
    recep_spec = mujoco.MjSpec.from_file(str(recep_xml))
    recep_body = recep_spec.worldbody.bodies[0]
    if not recep_body.first_joint():
        recep_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    rf = spec.worldbody.add_frame(pos=PROCTHOR_RECEPTACLE_POS, quat=THOR_QUAT)
    rf.attach_body(recep_body, "place_receptacle/", "")

    return spec, pickup_body.name


def build_custom_scene(robot_config, pickup_uid, receptacle_uid):
    """Build custom scene with primitive desk + bookcase."""
    spec = mujoco.MjSpec.from_file(str(HOUSE_BASE_XML))

    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.01], rgba=[0.3, 0.3, 0.3, 1.0], pos=[0, 0, 0], contype=8, conaffinity=15)

    # Desk
    desk = spec.worldbody.add_body(name="desk", pos=[0.75, 0.25, 0.0])
    desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5, 0.25, 0.015], pos=[0, 0, 0.72], rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)
    for lx, ly, ln in [(-0.45, -0.2, "fl"), (0.45, -0.2, "fr"), (-0.45, 0.2, "bl"), (0.45, 0.2, "br")]:
        desk.add_geom(name=f"desk_leg_{ln}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.36], pos=[lx, ly, 0.36], rgba=[0.55, 0.35, 0.2, 1.0], contype=8, conaffinity=15)

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
    robot_path = get_robot_path(robot_config.name) / robot_config.robot_xml_path
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    FrankaRobot.add_robot_to_scene(robot_config, spec, robot_spec,
        prefix=robot_config.robot_namespace, pos=[0, 0], quat=[1, 0, 0, 0])
    spec.camera(robot_config.robot_namespace + "gripper/wrist_camera").resolution = [RENDER_WIDTH, RENDER_HEIGHT]
    spec.body(robot_config.robot_namespace + "fr3_link0").add_camera(
        pos=[0.1, 0.57, 0.66], quat=[-0.3633, -0.1241, 0.4263, 0.8191],
        fovy=71.0, resolution=[RENDER_WIDTH, RENDER_HEIGHT], name="robot_0/exo_camera_1")

    # Pickup on desk
    pickup_xml = load_object(pickup_uid)
    pickup_spec = mujoco.MjSpec.from_file(str(pickup_xml))
    pickup_body = pickup_spec.worldbody.bodies[0]
    if not pickup_body.first_joint():
        pickup_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
    pf = spec.worldbody.add_frame(pos=[0.55, 0.25, CUSTOM_DESK_TOP_Z + 0.04], quat=THOR_QUAT)
    pf.attach_body(pickup_body, "pickup_object/", "")

    # Receptacle
    if receptacle_uid != "bookcase":
        recep_xml = load_object(receptacle_uid)
        recep_spec = mujoco.MjSpec.from_file(str(recep_xml))
        recep_body = recep_spec.worldbody.bodies[0]
        if not recep_body.first_joint():
            recep_body.add_joint(name="jntfree", type=mujoco.mjtJoint.mjJNT_FREE, damping=1.0)
        rf = spec.worldbody.add_frame(pos=[0.65, 0.35, CUSTOM_DESK_TOP_Z + 0.08], quat=THOR_QUAT)
        rf.attach_body(recep_body, "place_receptacle/", "")

    return spec, pickup_body.name


def run_single_episode(robot_config, policy, scene_type, pickup_uid, receptacle_uid, prompt, task_horizon, output_dir, episode_id):
    """Run a single episode. Returns success (bool) and video path."""
    if scene_type == "procthor":
        spec, pickup_body_name = build_procthor_scene(robot_config, pickup_uid, receptacle_uid)
    else:
        spec, pickup_body_name = build_custom_scene(robot_config, pickup_uid, receptacle_uid)

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

    # Record initial pickup position
    pickup_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object/" + pickup_body_name)
    initial_pickup_pos = data.xpos[pickup_body_id].copy() if pickup_body_id >= 0 else np.zeros(3)

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

    # Success heuristic: pickup object displaced significantly from start
    final_pickup_pos = data.xpos[pickup_body_id].copy() if pickup_body_id >= 0 else np.zeros(3)
    displacement = np.linalg.norm(final_pickup_pos - initial_pickup_pos)
    success = displacement > 0.05

    # Save video
    video_path = output_dir / f"{episode_id}.mp4"
    if frames:
        ImageSequenceClip(frames, fps=round(1000 / POLICY_DT_MS)).write_videofile(
            str(video_path), codec="libx264", audio=False, logger=None)

    renderer.close()
    return success, str(video_path)


def generate_report(results, output_dir, config):
    """Generate Markdown report and CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "pickup", "receptacle", "prompt", "group", "trial", "success", "video_path"])
        for r in results:
            writer.writerow([r["scene"], r["pickup"], r["receptacle"], r["prompt"],
                            r["group"], r["trial"], r["success"], r["video_path"]])

    # Aggregate by group
    group_stats = {}
    for r in results:
        g = r["group"]
        if g not in group_stats:
            group_stats[g] = {"success": 0, "total": 0}
        group_stats[g]["total"] += 1
        if r["success"]:
            group_stats[g]["success"] += 1

    # Per-combination stats
    combo_stats = {}
    for r in results:
        key = (r["scene"], r["pickup"], r["receptacle"])
        if key not in combo_stats:
            combo_stats[key] = {"success": 0, "total": 0, "group": r["group"], "prompt": r["prompt"]}
        combo_stats[key]["total"] += 1
        if r["success"]:
            combo_stats[key]["success"] += 1

    # Markdown report
    md_path = output_dir / f"report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark Report: Scene × Object Generalization\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Task horizon: {config['task_horizon']} steps\n")
        f.write(f"Repeats per combination: {config['repeats']}\n\n")

        f.write("## Summary by Group\n\n")
        f.write("| Group | Description | Success | Total | Rate |\n")
        f.write("|-------|-------------|---------|-------|------|\n")
        for g in sorted(group_stats.keys()):
            s = group_stats[g]
            rate = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
            f.write(f"| {g} | | {s['success']} | {s['total']} | {rate:.1f}% |\n")

        total_s = sum(s["success"] for s in group_stats.values())
        total_n = sum(s["total"] for s in group_stats.values())
        f.write(f"| **Total** | | **{total_s}** | **{total_n}** | **{total_s/total_n*100:.1f}%** |\n")

        f.write("\n## Scene Comparison\n\n")
        f.write("| Scene | Success | Total | Rate |\n")
        f.write("|-------|---------|-------|------|\n")
        for scene in ["procthor", "custom"]:
            sr = [r for r in results if r["scene"] == scene]
            if sr:
                s = sum(1 for r in sr if r["success"])
                f.write(f"| {scene} | {s} | {len(sr)} | {s/len(sr)*100:.1f}% |\n")

        f.write("\n## Object Type Comparison\n\n")
        f.write("| Object Type | Success | Total | Rate |\n")
        f.write("|-------------|---------|-------|------|\n")
        for otype in ["thor", "objaverse"]:
            sr = [r for r in results if get_object_type(r["receptacle"]) == otype]
            if sr:
                s = sum(1 for r in sr if r["success"])
                f.write(f"| {otype} | {s} | {len(sr)} | {s/len(sr)*100:.1f}% |\n")

        f.write("\n## Per-Combination Results\n\n")
        f.write("| Scene | Pickup | Receptacle | Group | Success | Rate |\n")
        f.write("|-------|--------|-----------|-------|---------|------|\n")
        for (scene, pickup, receptacle), stats in sorted(combo_stats.items()):
            rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            pickup_name = get_object_name(pickup)
            recep_name = get_object_name(receptacle)
            f.write(f"| {scene} | {pickup_name} | {recep_name} | {stats['group'][:1]} | {stats['success']}/{stats['total']} | {rate:.0f}% |\n")

        f.write(f"\n## Data\n\n")
        f.write(f"- CSV: `{csv_path.name}`\n")
        f.write(f"- Videos: `{output_dir}/`\n")

    print(f"\nReport: {md_path}")
    print(f"CSV:    {csv_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for g in sorted(group_stats.keys()):
        s = group_stats[g]
        rate = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"{g}: {s['success']}/{s['total']} ({rate:.1f}%)")
    print(f"{'='*60}")
    print(f"Total: {total_s}/{total_n} ({total_s/total_n*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation with scene × object generalization report")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--config", type=str, help="Path to batch config JSON")
    parser.add_argument("--generate-config", action="store_true", help="Generate default config and exit")
    parser.add_argument("--task_horizon_override", type=int, default=None)
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else BENCHMARK_DIR / f"batch_results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Pre-download objects
    print("\nPre-downloading objects...")
    all_objects = set()
    for task in tasks:
        all_objects.add(task["pickup"])
        if task.get("receptacle", "bookcase") != "bookcase":
            all_objects.add(task["receptacle"])

    for obj_uid in sorted(all_objects):
        try:
            load_object(obj_uid)
            print(f"  OK: {obj_uid}", flush=True)
        except Exception as e:
            print(f"  FAIL: {obj_uid} — {e}", flush=True)

    # Load policy once
    print("\nLoading policy (one-time)...", flush=True)
    from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig

    policy_config = RealRobotVLAPolicyConfig()
    policy_config.checkpoint_path = args.checkpoint_path
    policy_config.action_type = "joint_pos"
    policy_config.action_keys["arm"] = "joint_pos"

    class MockConfig:
        def __init__(self, pc):
            self.policy_config = pc

    policy = RealRobotVLAPolicy(config=MockConfig(policy_config), task_type="manipulation")
    robot_config = FrankaRobotConfig(base_size=[0.5, 0.5, 0.75])
    print("Policy loaded.\n", flush=True)

    # Run episodes
    results = []
    episode_num = 0
    for task_idx, task in enumerate(tasks):
        scene_type = task.get("scene", "custom")
        pickup = task["pickup"]
        receptacle = task.get("receptacle", "bookcase")

        pickup_name = get_object_name(pickup)
        receptacle_name = get_object_name(receptacle)
        prompt = task.get("prompt") or f"pick up the {pickup_name} and place it in the {receptacle_name}"
        group = get_group_name(scene_type, receptacle)

        for trial in range(repeats):
            episode_num += 1
            episode_id = f"ep{episode_num:03d}_{scene_type}_{pickup_name}_{receptacle_name}_t{trial}"
            print(f"[{episode_num}/{total_episodes}] [{scene_type}] {pickup_name} → {receptacle_name} (trial {trial+1}/{repeats})", flush=True)

            try:
                success, video_path = run_single_episode(
                    robot_config, policy, scene_type,
                    pickup, receptacle, prompt,
                    task_horizon, output_dir, episode_id,
                )
                status = "PASS" if success else "FAIL"
                print(f"  → {status}", flush=True)
            except Exception as e:
                success = False
                video_path = ""
                print(f"  → ERROR: {e}", flush=True)

            results.append({
                "scene": scene_type,
                "pickup": pickup,
                "receptacle": receptacle,
                "prompt": prompt,
                "group": group,
                "trial": trial,
                "success": success,
                "video_path": video_path,
            })

            # Save partial results
            with open(output_dir / "results_partial.json", "w") as f:
                json.dump({"completed": episode_num, "total": total_episodes, "results": results}, f, indent=2)

    # Final report
    generate_report(results, output_dir, config)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    partial = output_dir / "results_partial.json"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
