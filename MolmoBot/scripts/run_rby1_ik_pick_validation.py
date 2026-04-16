#!/usr/bin/env python3
"""Validate RBY1 salt-shaker pickability using MuJoCo IK (no CuRobo / no CUDA).

This script computes the CORRECT grasp pose for the salt shaker using the
MuJoCo-based damped least squares IK solver from molmospaces, then runs the
scripted grasp eval to test whether the simulator physics can execute the pick.

Unlike the previous scripted grasp sanity (which replayed misaligned policy
waypoints), this script computes a fresh IK solution that places the finger
midpoint directly on the object center.

Usage:
    .venv/bin/python scripts/run_rby1_ik_pick_validation.py
    .venv/bin/python scripts/run_rby1_ik_pick_validation.py --task_horizon 400
    .venv/bin/python scripts/run_rby1_ik_pick_validation.py --approach_height 0.12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rby1_adaptive_variants import (
    DEFAULT_CHECKPOINT,
    RunRecord,
    Variant,
    create_benchmark,
    eval_command,
    find_latest_h5,
    find_videos,
    make_run_record,
    run_blocking as _run_blocking_orig,
    validate_smoke,
)


def _platform_eval_env() -> dict[str, str]:
    """Return eval environment adapted for the current platform."""
    env = os.environ.copy()
    env.setdefault("MLSPACES_RENDER_DEVICE_ID", "none")
    if sys.platform == "darwin":
        # macOS: osmesa is not available; use default GL backend
        env.pop("MUJOCO_GL", None)
        env.pop("PYOPENGL_PLATFORM", None)
    else:
        env.setdefault("MUJOCO_GL", "osmesa")
        env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    return env


def run_blocking(record: RunRecord) -> RunRecord:
    """Platform-aware wrapper for run_blocking."""
    import subprocess, time as _time
    record.started_at = _time.time()
    log_path = Path(record.output_dir) / f"{record.stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record.log_path = str(log_path)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            record.command,
            cwd=REPO_ROOT,
            env=_platform_eval_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record.returncode = proc.returncode
    record.finished_at = _time.time()
    return record
from scripts.run_rby1_privileged_grasp_sanity import (
    BEST_OBJECT_POSE,
    SCRIPTED_CONFIG,
    enrich_summary,
    score_summary,
    scripted_config,
)

# ── IK imports from molmospaces ──────────────────────────────────────────

from molmo_spaces.kinematics.rby1_kinematics import RBY1Kinematics
from molmo_spaces.robots.robot_views.rby1_view import RBY1RobotView

# ── Constants ────────────────────────────────────────────────────────────

OUTPUT_ROOT = REPO_ROOT / "eval_output" / "rby1_ik_pick_validation"

# Salt shaker object pose from benchmark episode index 5 (best previous result)
DEFAULT_OBJECT_POS = np.array([0.445, 0.28, 0.805])

# EE site to finger midpoint offset in the wrist frame (from MJCF analysis):
# ee_site_l is at (0, 0, -0.2572) in link_left_arm_6 frame
# finger midpoint is at (0, 0, -0.2102) in link_left_arm_6 frame
# offset = (0, 0, +0.047) in wrist local frame
# When the EE frame Z points downward (toward table), moving the EE 47mm
# "backward" (up, away from table) places the fingertips on the target.
EE_TO_FINGERTIP_OFFSET_LOCAL_Z = 0.047  # meters


def find_rby1_model_path() -> Path:
    """Find the RBY1 MJCF model from the installed molmospaces assets."""
    import molmo_spaces

    pkg_root = Path(molmo_spaces.__file__).parent
    # Check several possible asset locations
    candidates = [
        pkg_root / "assets" / "robots" / "rby1m" / "rby1_v1.2_site_control.xml",
        pkg_root.parent / "assets" / "robots" / "rby1m" / "rby1_v1.2_site_control.xml",
    ]
    # Also check in the installed package's site-packages
    for site_pkg in sys.path:
        p = Path(site_pkg) / "assets" / "robots" / "rby1m" / "rby1_v1.2_site_control.xml"
        if p.exists():
            return p

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find RBY1 MJCF model. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def load_custom_scene_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the custom salt-shaker scene with the RBY1 robot."""
    scene_xml = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark" / "custom_scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(f"Custom scene not found: {scene_xml}")

    # The scene XML references robot and object assets via relative paths.
    # We need to find the correct asset directory.
    import molmo_spaces.molmo_spaces_constants as constants

    assets_dir = constants.ASSETS_DIR
    model = mujoco.MjModel.from_xml_path(
        str(scene_xml),
        {str(scene_xml.parent): str(assets_dir)},
    )
    data = mujoco.MjData(model)
    return model, data


def load_standalone_rby1_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load standalone RBY1 model for IK computation."""
    model_path = find_rby1_model_path()
    cwd = os.getcwd()
    os.chdir(model_path.parent)
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(model)
    return model, data


def set_initial_qpos(robot_view: RBY1RobotView, benchmark_qpos: dict) -> None:
    """Set the robot to the initial configuration from the benchmark."""
    qpos_dict = {}
    for group_name in robot_view.move_group_ids():
        if group_name in benchmark_qpos:
            qpos_dict[group_name] = np.array(benchmark_qpos[group_name])
        else:
            qpos_dict[group_name] = robot_view.get_move_group(group_name).joint_pos.copy()
    robot_view.set_qpos_dict(qpos_dict)


def compute_top_down_grasp_pose(
    object_pos: np.ndarray,
    ee_to_fingertip_z: float = EE_TO_FINGERTIP_OFFSET_LOCAL_Z,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Compute the EE target pose for a top-down grasp centered on the object.

    Args:
        object_pos: XYZ position of the object center.
        ee_to_fingertip_z: Offset from EE site to finger midpoint along wrist Z.
        rotation_deg: Rotation of the finger opening axis around world Z (degrees).
            0 = fingers open along world Y. 30 = rotated 30 deg CCW, etc.
    """
    angle_rad = np.radians(rotation_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    pose = np.eye(4)
    # Rotation: wrist local Z → world -Z (down), local X (finger opening) rotated
    pose[:3, :3] = np.array([
        [-s, c, 0],
        [c, s, 0],
        [0, 0, -1],
    ])

    # EE must be BELOW object by the offset so finger midpoint lands at object center
    pose[:3, 3] = object_pos.copy()
    pose[2, 3] -= ee_to_fingertip_z

    return pose


def gravity_settle(model, data, robot_view, n_steps=500):
    """Run simulation forward to let gravity settle, then return settled qpos."""
    # Set ctrl to current qpos so the PD controller tries to hold position
    for i in range(model.nu):
        data.ctrl[i] = 0.0  # Will be overridden by actuator dynamics

    # Actually, we need to set the actuator targets to current positions
    # to hold the robot in place during settling
    for group_name in robot_view.move_group_ids():
        mg = robot_view.get_move_group(group_name)
        if hasattr(mg, 'ctrl') and hasattr(mg, 'joint_pos'):
            try:
                mg.ctrl = mg.joint_pos.copy()
            except Exception:
                pass

    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def solve_ik_waypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_pos: np.ndarray,
    approach_height: float = 0.12,
    lift_height: float = 0.15,
    arm_only: bool = False,
) -> dict[str, dict[str, list[float]]] | None:
    """Solve IK for approach, grasp, and lift waypoints.

    Computes IK after gravity settling so the solution is near gravitational
    equilibrium and the PD controller can maintain it.

    Returns dict of {phase: {move_group: joint_positions}} or None if IK fails.
    """
    robot_view = RBY1RobotView(data, namespace="robot_0/", holo_base=True)
    kinematics = RBY1Kinematics(model, data, namespace="robot_0/", holo_base=True)

    # Set initial robot configuration from benchmark
    benchmark_path = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark" / "benchmark.json"
    with open(benchmark_path) as f:
        episodes = json.load(f)
    init_qpos = episodes[-1]["robot"]["init_qpos"]  # episode 5 (best previous)
    set_initial_qpos(robot_view, init_qpos)
    mujoco.mj_forward(model, data)

    # Let gravity settle to find the stable equilibrium
    print("  Settling gravity (500 sim steps)...")
    gravity_settle(model, data, robot_view, n_steps=500)
    settled_qpos = robot_view.get_qpos_dict()
    print(f"  Settled torso: {[round(v, 3) for v in settled_qpos['torso']]}")
    print(f"  (Original:     {init_qpos['torso']})")

    base_pose = robot_view.base.pose
    q0 = settled_qpos  # Use gravity-settled state as IK starting point

    unlocked_groups = ["left_arm"] if arm_only else ["torso", "left_arm"]
    print(f"  IK mode: {'arm-only (torso locked)' if arm_only else 'torso + arm'}")

    # 1. Compute grasp pose - try multiple orientations in arm-only mode
    orientations = [0, 15, 30, -30, 60, -60, 90, -90] if arm_only else [0]
    grasp_pose = None
    grasp_qpos = None
    best_orient = None

    for orient_deg in orientations:
        candidate_pose = compute_top_down_grasp_pose(object_pos, rotation_deg=orient_deg)

        robot_view.set_qpos_dict(settled_qpos)
        mujoco.mj_forward(model, data)

        candidate_qpos = kinematics.ik(
            move_group_id="left_arm",
            pose=candidate_pose,
            unlocked_move_group_ids=unlocked_groups,
            q0=robot_view.get_qpos_dict(),
            base_pose=base_pose,
            rel_to_base=False,
            eps=1e-3,
            max_iter=2000,
            dt=0.5,
        )
        if candidate_qpos is not None:
            grasp_pose = candidate_pose
            grasp_qpos = candidate_qpos
            best_orient = orient_deg
            break
        elif arm_only:
            print(f"  [{orient_deg:+d}deg] grasp IK failed, trying next...")

    if grasp_qpos is None:
        print("  [FAIL] IK failed for grasp pose (all orientations)")
        return None

    print(f"\n  Grasp target EE pose (orient={best_orient:+d}deg):\n{grasp_pose}")
    print(f"  Object pos: {object_pos}")
    print(f"  EE target pos: {grasp_pose[:3, 3]} (offset by {EE_TO_FINGERTIP_OFFSET_LOCAL_Z}m)")
    print(f"  [OK] Grasp IK converged")
    print(f"  Grasp torso: {[round(v, 3) for v in grasp_qpos['torso']]}")

    # Verify EE position after IK
    robot_view.set_qpos_dict(grasp_qpos)
    mujoco.mj_forward(model, data)
    left_arm = robot_view.get_move_group("left_arm")
    actual_ee = left_arm.leaf_frame_to_world[:3, 3]
    print(f"  Actual EE pos after IK: {actual_ee}")
    print(f"  EE error: {np.linalg.norm(actual_ee - grasp_pose[:3, 3]):.6f} m")

    # 3. Compute approach pose (same as grasp but higher)
    approach_pose = grasp_pose.copy()
    approach_pose[2, 3] += approach_height
    print(f"\n  Approach target pos: {approach_pose[:3, 3]}")

    # Reset to settled config for approach IK
    robot_view.set_qpos_dict(settled_qpos)
    mujoco.mj_forward(model, data)

    approach_qpos = kinematics.ik(
        move_group_id="left_arm",
        pose=approach_pose,
        unlocked_move_group_ids=unlocked_groups,
        q0=settled_qpos,
        base_pose=base_pose,
        rel_to_base=False,
        eps=1e-3,
        max_iter=2000,
        dt=0.5,
    )
    if approach_qpos is None:
        print("  [FAIL] IK failed for approach pose")
        return None
    print(f"  [OK] Approach IK converged")
    print(f"  Approach torso: {[round(v, 3) for v in approach_qpos['torso']]}")

    # 4. Compute lift pose (same as grasp but higher)
    lift_pose = grasp_pose.copy()
    lift_pose[2, 3] += lift_height
    print(f"\n  Lift target pos: {lift_pose[:3, 3]}")

    # Use grasp config as starting point for lift IK
    lift_qpos = kinematics.ik(
        move_group_id="left_arm",
        pose=lift_pose,
        unlocked_move_group_ids=unlocked_groups,
        q0=grasp_qpos,
        base_pose=base_pose,
        rel_to_base=False,
        eps=1e-3,
        max_iter=2000,
        dt=0.5,
    )
    if lift_qpos is None:
        print("  [FAIL] IK failed for lift pose")
        return None
    print(f"  [OK] Lift IK converged")

    # 5. Extract the relevant move groups
    targets = {
        "approach": {
            k: v.tolist() for k, v in approach_qpos.items()
            if k in ("base", "torso", "left_arm")
        },
        "grasp": {
            k: v.tolist() for k, v in grasp_qpos.items()
            if k in ("base", "torso", "left_arm")
        },
        "lift": {
            k: v.tolist() for k, v in lift_qpos.items()
            if k in ("base", "torso", "left_arm")
        },
    }
    return targets


def solve_with_orientation_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_pos: np.ndarray,
    approach_height: float = 0.12,
    lift_height: float = 0.15,
) -> list[tuple[str, dict[str, dict[str, list[float]]]]]:
    """Try multiple grasp orientations and return all successful IK solutions."""
    robot_view = RBY1RobotView(data, namespace="robot_0/", holo_base=True)
    kinematics = RBY1Kinematics(model, data, namespace="robot_0/", holo_base=True)

    benchmark_path = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark" / "benchmark.json"
    with open(benchmark_path) as f:
        episodes = json.load(f)
    init_qpos = episodes[-1]["robot"]["init_qpos"]

    results: list[tuple[str, dict]] = []

    # Try several orientations: rotate the finger opening axis around Z
    # Angles represent rotation of the finger opening axis from world Y
    for angle_deg in [0, 30, -30, 45, -45, 60, -60, 90]:
        name = f"orient_{angle_deg:+d}deg"
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Base top-down orientation rotated around world Z
        # col0 (local X = finger opening): rotated in XY plane from world Y
        # col1 (local Y): perpendicular in XY plane
        # col2 (local Z = toward fingertips): world -Z (pointing down)
        R = np.array([
            [-s, c, 0],
            [c, s, 0],
            [0, 0, -1],
        ])

        grasp_pose = np.eye(4)
        grasp_pose[:3, :3] = R
        grasp_pose[:3, 3] = object_pos.copy()
        grasp_pose[2, 3] -= EE_TO_FINGERTIP_OFFSET_LOCAL_Z  # EE below object

        approach_pose = grasp_pose.copy()
        approach_pose[2, 3] += approach_height

        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += lift_height

        unlocked_groups = ["torso", "left_arm"]

        # Grasp IK
        set_initial_qpos(robot_view, init_qpos)
        mujoco.mj_forward(model, data)
        q0 = robot_view.get_qpos_dict()
        base_pose = robot_view.base.pose

        grasp_qpos = kinematics.ik(
            "left_arm", grasp_pose, unlocked_groups, q0, base_pose,
            eps=1e-3, max_iter=2000, dt=0.5,
        )
        if grasp_qpos is None:
            print(f"  [{name}] grasp IK FAILED")
            continue

        # Approach IK
        set_initial_qpos(robot_view, init_qpos)
        mujoco.mj_forward(model, data)
        q0 = robot_view.get_qpos_dict()

        approach_qpos = kinematics.ik(
            "left_arm", approach_pose, unlocked_groups, q0, base_pose,
            eps=1e-3, max_iter=2000, dt=0.5,
        )
        if approach_qpos is None:
            print(f"  [{name}] approach IK FAILED")
            continue

        # Lift IK
        lift_qpos = kinematics.ik(
            "left_arm", lift_pose, unlocked_groups, grasp_qpos, base_pose,
            eps=1e-3, max_iter=2000, dt=0.5,
        )
        if lift_qpos is None:
            print(f"  [{name}] lift IK FAILED")
            continue

        targets = {
            "approach": {
                k: v.tolist() for k, v in approach_qpos.items()
                if k in ("base", "torso", "left_arm")
            },
            "grasp": {
                k: v.tolist() for k, v in grasp_qpos.items()
                if k in ("base", "torso", "left_arm")
            },
            "lift": {
                k: v.tolist() for k, v in lift_qpos.items()
                if k in ("base", "torso", "left_arm")
            },
        }
        results.append((name, targets))
        print(f"  [{name}] ALL IK OK")

    return results


def report_path(output_root: Path, suffix: str) -> Path:
    return output_root / f"ik_pick_validation_report.{suffix}"


def write_report(output_root: Path, records: list[RunRecord], ik_metadata: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "ik_metadata": ik_metadata,
        "records": [asdict(record) for record in records],
    }
    report_path(output_root, "json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# RBY1 IK Pick Validation",
        "",
        "This run uses MuJoCo IK (no CuRobo) to compute the CORRECT grasp",
        "waypoints, then executes via the scripted grasp eval infrastructure.",
        "",
        f"- object pos: `{ik_metadata.get('object_pos')}`",
        f"- approach height: `{ik_metadata.get('approach_height')}`m",
        f"- ee-to-fingertip offset: `{EE_TO_FINGERTIP_OFFSET_LOCAL_Z}`m",
        "",
        "## Runs",
    ]
    for record in records:
        summary = record.summary or {}
        diagnostic = summary.get("diagnostic") or {}
        lines.append(
            f"- {record.variant.name if hasattr(record.variant, 'name') else record.variant}: "
            f"rc={record.returncode} "
            f"physical_grasp_success={summary.get('physical_grasp_success')} "
            f"obj_delta={diagnostic.get('obj_delta_m')} "
            f"z_lift_max={diagnostic.get('obj_z_lift_max_m')} "
            f"left_mid_min={diagnostic.get('left_midpoint_min_m')} "
            f"h5={record.h5_path}"
        )
    report_path(output_root, "md").write_text("\n".join(lines) + "\n")


def patch_benchmark_init_qpos(benchmark_dir: Path, approach_qpos: dict) -> None:
    """Patch the benchmark's init_qpos so the robot starts at the IK approach pose.

    This eliminates the large torso delta that the PD controller can't track
    due to gravity-induced steady-state error.
    """
    bm_path = benchmark_dir / "benchmark.json"
    episodes = json.loads(bm_path.read_text())
    for ep in episodes:
        init = ep["robot"]["init_qpos"]
        if "torso" in approach_qpos:
            init["torso"] = [float(v) for v in approach_qpos["torso"]]
        if "left_arm" in approach_qpos:
            init["left_arm"] = [float(v) for v in approach_qpos["left_arm"]]
        # Keep base at origin, don't change head/grippers
    bm_path.write_text(json.dumps(episodes, indent=2) + "\n")


def run_variant(
    name: str,
    targets: dict,
    output_root: Path,
    checkpoint_path: Path,
    task_horizon: int,
    object_pose: list,
    arm_only: bool = False,
) -> RunRecord:
    """Run a single IK variant through the scripted grasp eval."""
    variant = Variant(
        name=name,
        object_pose=object_pose,
        eval_config_cls=SCRIPTED_CONFIG,
        mode="ik_scripted",
        reason=f"IK-computed grasp: {name}",
    )

    # Phase steps: approach is now short since we start pre-positioned.
    # Grasp descent is the main phase that needs enough steps.
    ik_phase_steps = {
        "open": 10,
        "approach": 20,   # small delta from pre-positioned state
        "grasp": 60,      # descent to object - needs convergence time
        "close": 50,      # close gripper firmly
        "lift": 120,      # lift with object
        "hold": 40,       # hold at top
    }
    payload = scripted_config(targets, ik_phase_steps)
    payload["gain_multiplier"] = 100.0  # Boost PD gains heavily for gravity compensation

    benchmark_dir = output_root / name / "ik_scripted" / "benchmark"
    create_benchmark(variant, benchmark_dir)

    # Pre-position the robot at the IK approach solution so the controller
    # only needs to handle the small delta to grasp/lift.
    # This also prevents the arm from sweeping through the object during
    # the large joint motion from default init to approach target.
    approach_qpos = targets.get("approach", {})
    if approach_qpos:
        patch_benchmark_init_qpos(benchmark_dir, approach_qpos)

    output_dir = output_root / name / "ik_scripted" / "full"
    record = make_run_record(variant, "full", benchmark_dir, output_dir, checkpoint_path, task_horizon)
    record.command = [
        "env",
        f"RBY1_SCRIPTED_GRASP_CONFIG_JSON={json.dumps(payload)}",
        *record.command,
    ]
    record = run_blocking(record)

    h5_path = find_latest_h5(output_dir)
    if h5_path is not None:
        record.h5_path = str(h5_path)
        record.videos = find_videos(output_dir)
        record.summary = enrich_summary(h5_path)
        record.score = score_summary(record.summary)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="RBY1 IK pick validation (no CuRobo)")
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task_horizon", type=int, default=400)
    parser.add_argument("--approach_height", type=float, default=0.12,
                        help="Height above object for approach waypoint (meters)")
    parser.add_argument("--lift_height", type=float, default=0.15,
                        help="Height above object for lift waypoint (meters)")
    parser.add_argument("--object_pos", type=float, nargs=3,
                        default=list(DEFAULT_OBJECT_POS),
                        help="Salt shaker XYZ position")
    parser.add_argument("--single_orientation", action="store_true",
                        help="Only try the default orientation (skip sweep)")
    parser.add_argument("--arm_only", action="store_true",
                        help="Lock torso at zero, only solve IK for arm joints")
    args = parser.parse_args()

    object_pos = np.array(args.object_pos)
    object_pose = list(BEST_OBJECT_POSE)
    object_pose[:3] = object_pos.tolist()

    print("=" * 60)
    print("RBY1 IK Pick Validation")
    print("=" * 60)
    print(f"Object position: {object_pos}")
    print(f"Approach height: {args.approach_height}m")
    print(f"Lift height: {args.lift_height}m")
    print(f"EE-to-fingertip offset: {EE_TO_FINGERTIP_OFFSET_LOCAL_Z}m")
    print(f"Arm-only mode: {args.arm_only}")
    print()

    # Step 1: Load model and solve IK
    print("[1/3] Loading RBY1 model...")
    model, data = load_standalone_rby1_model()

    print("[2/3] Solving IK for grasp waypoints...")
    if args.single_orientation:
        targets = solve_ik_waypoints(
            model, data, object_pos, args.approach_height, args.lift_height,
            arm_only=args.arm_only,
        )
        if targets is None:
            print("\n[ABORT] IK failed for the default orientation.")
            print("Try --approach_height or --object_pos adjustments.")
            return
        ik_solutions = [("default_topdown", targets)]
    else:
        ik_solutions = solve_with_orientation_sweep(
            model, data, object_pos, args.approach_height, args.lift_height
        )
        if not ik_solutions:
            print("\n[ABORT] IK failed for all orientations.")
            print("The object may be unreachable from the current robot position.")
            return

    print(f"\n  {len(ik_solutions)} orientation(s) solved successfully")
    for name, _ in ik_solutions:
        print(f"    - {name}")

    # Step 2: Run eval for each successful IK solution
    print(f"\n[3/3] Running scripted eval for {len(ik_solutions)} variant(s)...")
    ik_metadata = {
        "object_pos": object_pos.tolist(),
        "approach_height": args.approach_height,
        "lift_height": args.lift_height,
        "ee_to_fingertip_offset_z": EE_TO_FINGERTIP_OFFSET_LOCAL_Z,
        "num_orientations_tried": 8 if not args.single_orientation else 1,
        "num_ik_successes": len(ik_solutions),
    }
    records: list[RunRecord] = []

    for name, targets in ik_solutions:
        print(f"\n  Running {name}...")
        record = run_variant(
            name, targets, args.output_root, args.checkpoint_path,
            args.task_horizon, object_pose, arm_only=args.arm_only,
        )
        records.append(record)
        write_report(args.output_root, records, ik_metadata)

        summary = record.summary or {}
        diagnostic = summary.get("diagnostic") or {}
        success = summary.get("physical_grasp_success", False)
        print(
            f"  [{name}] "
            f"physical_grasp_success={success} "
            f"obj_delta={diagnostic.get('obj_delta_m')} "
            f"z_lift_max={diagnostic.get('obj_z_lift_max_m')} "
            f"left_mid_min={diagnostic.get('left_midpoint_min_m')}"
        )
        if success:
            print(f"\n  *** SUCCESS! {name} picked up the salt shaker! ***")
            break

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    any_success = any(
        (r.summary or {}).get("physical_grasp_success") for r in records
    )
    if any_success:
        print("At least one IK orientation SUCCEEDED in picking up the salt shaker.")
        print("This confirms the physics are fine - the failure is purely policy alignment.")
        print("Next step: fine-tune the checkpoint with correctly-aligned demonstrations.")
    else:
        print("All IK orientations FAILED to pick up the salt shaker.")
        print("This suggests a physics/scene issue, not just policy alignment.")
        print("Next: check collision mesh, friction, contact geometry.")

    best = max(records, key=lambda r: r.score or -1.0)
    best_summary = best.summary or {}
    print(f"\nBest variant: {best.variant.name if hasattr(best.variant, 'name') else best.variant}")
    print(f"  score: {best.score}")
    print(f"  physical_grasp_success: {best_summary.get('physical_grasp_success')}")
    print(f"  obj_delta: {best_summary.get('diagnostic', {}).get('obj_delta_m')}")
    print(f"  z_lift_max: {best_summary.get('diagnostic', {}).get('obj_z_lift_max_m')}")
    print(f"  left_midpoint_min: {best_summary.get('diagnostic', {}).get('left_midpoint_min_m')}")
    if best.h5_path:
        print(f"  h5: {best.h5_path}")
    if best.videos:
        print(f"  video: {best.videos[0]}")
    print(f"\nReport: {report_path(args.output_root, 'md')}")


if __name__ == "__main__":
    main()
