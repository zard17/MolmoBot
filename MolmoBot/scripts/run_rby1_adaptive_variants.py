#!/usr/bin/env python3
"""Run an adaptive RBY1 salt-shaker geometry search.

The runner executes a ranked list of one-episode benchmark variants. Each
variant gets a short smoke eval before a full eval. Results are summarized after
each full run and used to add targeted follow-up variants.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generate_rby1_pickpnp_benchmark import (
    EPISODE_CONFIGS,
    ROBOT_BASE_POSE,
    SALT_SHAKER_BODY_NAME,
    create_rby1_episode,
)
from scripts.summarize_rby1_eval import summarize_h5, task_info_at


SOURCE_BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark"
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpts" / "molmobot" / "MolmoBot-RBY1Multitask"

FROZEN_CONFIG = "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPFrozenBaseEvalConfig"
DEFAULT_CONFIG = "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig"
CONTACT_HOLD_CONFIG = (
    "olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPContactHoldFrozenBaseEvalConfig"
)

BASE_OBJECT_POSE = [0.452, 0.268, 0.807, 0.7071068, 0.7071068, 0.0, 0.0]
OBJECT_MOVE_THRESHOLD_M = 0.005
PROMISING_MIN_DIST_M = 0.01
PROMISING_CONTACT_MIN_DIST_M = 0.02
CONTACT_Z_MARKER = "_contact_z_"
TOWARD_TCP_MARKER = "_toward_tcp_"


@dataclass
class Variant:
    name: str
    object_pose: list[float] = field(default_factory=lambda: list(BASE_OBJECT_POSE))
    robot_base_pose: list[float] = field(default_factory=lambda: list(ROBOT_BASE_POSE))
    scene_table_yaw_rad: float | None = None
    eval_config_cls: str = FROZEN_CONFIG
    mode: str = "frozen"
    priority: float = 0.0
    parent: str | None = None
    reason: str = ""


@dataclass
class RunRecord:
    variant: dict[str, Any]
    stage: str
    output_dir: str
    benchmark_dir: str
    command: list[str]
    returncode: int | None = None
    log_path: str | None = None
    h5_path: str | None = None
    videos: list[str] = field(default_factory=list)
    smoke_ok: bool | None = None
    smoke_reason: str | None = None
    summary: dict[str, Any] | None = None
    score: float | None = None
    started_at: float | None = None
    finished_at: float | None = None


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def yaw_quat(yaw_rad: float) -> list[float]:
    return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]


def with_object_yaw(pose: list[float], yaw_deg: float) -> list[float]:
    adjusted = list(pose)
    adjusted[3:7] = quat_multiply(yaw_quat(math.radians(yaw_deg)), adjusted[3:7])
    return adjusted


def with_robot_yaw(yaw_deg: float) -> list[float]:
    pose = list(ROBOT_BASE_POSE)
    pose[3:7] = yaw_quat(math.radians(yaw_deg))
    return pose


def with_robot_y(offset_y: float) -> list[float]:
    pose = list(ROBOT_BASE_POSE)
    pose[1] += offset_y
    return pose


def initial_variants() -> list[Variant]:
    base = list(BASE_OBJECT_POSE)
    return [
        Variant("baseline_best_pose", base, priority=100, reason="Best measured pose."),
        Variant(
            "object_y_minus_015",
            [base[0], base[1] - 0.015, base[2], *base[3:]],
            priority=90,
            reason="Lateral centering sweep.",
        ),
        Variant(
            "object_y_plus_015",
            [base[0], base[1] + 0.015, base[2], *base[3:]],
            priority=89,
            reason="Opposite lateral centering sweep.",
        ),
        Variant(
            "object_x_minus_015",
            [base[0] - 0.015, base[1], base[2], *base[3:]],
            priority=80,
            reason="Move object closer to robot.",
        ),
        Variant(
            "object_x_plus_015",
            [base[0] + 0.015, base[1], base[2], *base[3:]],
            priority=79,
            reason="Move object farther along approach path.",
        ),
        Variant(
            "object_z_plus_015",
            [base[0], base[1], base[2] + 0.015, *base[3:]],
            priority=70,
            reason="Raise object to reduce table/finger interference.",
        ),
        Variant(
            "object_z_minus_010",
            [base[0], base[1], base[2] - 0.010, *base[3:]],
            priority=69,
            reason="Lower object to test learned grasp height.",
        ),
        Variant(
            "robot_yaw_minus_10deg",
            base,
            robot_base_pose=with_robot_yaw(-10.0),
            priority=60,
            reason="Rotate robot base approach angle.",
        ),
        Variant(
            "robot_yaw_plus_10deg",
            base,
            robot_base_pose=with_robot_yaw(10.0),
            priority=59,
            reason="Rotate robot base approach angle.",
        ),
        Variant(
            "robot_y_plus_030",
            base,
            robot_base_pose=with_robot_y(0.03),
            priority=50,
            reason="Small lateral robot-base offset.",
        ),
        Variant(
            "robot_y_minus_030",
            base,
            robot_base_pose=with_robot_y(-0.03),
            priority=49,
            reason="Small lateral robot-base offset.",
        ),
        Variant(
            "object_orientation_yaw_90",
            with_object_yaw(base, 90.0),
            priority=40,
            reason="Expose a different salt-shaker collision side.",
        ),
        Variant(
            "object_orientation_yaw_minus_90",
            with_object_yaw(base, -90.0),
            priority=39,
            reason="Expose a different salt-shaker collision side.",
        ),
        Variant(
            "table_rotation_plus_5deg",
            base,
            scene_table_yaw_rad=math.radians(5.0),
            priority=20,
            reason="Small table yaw scene variant.",
        ),
        Variant(
            "table_rotation_minus_5deg",
            base,
            scene_table_yaw_rad=math.radians(-5.0),
            priority=19,
            reason="Small table yaw scene variant.",
        ),
    ]


def create_benchmark(variant: Variant, benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for name in ("custom_scene.xml", "custom_scene_metadata.json"):
        source = SOURCE_BENCHMARK_DIR / name
        if source.is_file():
            shutil.copyfile(source, benchmark_dir / name)

    if variant.scene_table_yaw_rad is not None:
        scene_path = benchmark_dir / "custom_scene.xml"
        tree = ET.parse(scene_path)
        root = tree.getroot()
        for body in root.findall(".//body"):
            if body.attrib.get("name") == "desk":
                body.set("euler", f"0 0 {variant.scene_table_yaw_rad:.8f}")
                break
        tree.write(scene_path, encoding="unicode")

    config = copy.deepcopy(EPISODE_CONFIGS[-1])
    config["pickup_pos"] = list(variant.object_pose)
    config["goal_pos"] = list(variant.object_pose)
    config["goal_pos"][2] += 0.2
    episode = create_rby1_episode(0, config)
    episode["task"]["robot_base_pose"] = list(variant.robot_base_pose)
    episode["task"]["pickup_obj_start_pose"] = list(variant.object_pose)
    episode["task"]["pickup_obj_goal_pose"] = list(config["goal_pos"])
    episode["scene_modifications"]["object_poses"][SALT_SHAKER_BODY_NAME] = list(
        variant.object_pose
    )
    (benchmark_dir / "benchmark.json").write_text(json.dumps([episode], indent=2) + "\n")


def eval_command(
    checkpoint_path: Path,
    benchmark_dir: Path,
    eval_config_cls: str,
    task_horizon: int,
    output_dir: Path,
    num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "launch_scripts/run_eval.py",
        "--checkpoint_path",
        str(checkpoint_path),
        "--benchmark_path",
        str(benchmark_dir),
        "--eval_config_cls",
        eval_config_cls,
        "--task_horizon",
        str(task_horizon),
        "--output_dir",
        str(output_dir),
        "--num_workers",
        str(num_workers),
    ]


def eval_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MLSPACES_RENDER_DEVICE_ID", "none")
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    return env


def run_blocking(record: RunRecord) -> RunRecord:
    record.started_at = time.time()
    log_path = Path(record.output_dir) / f"{record.stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record.log_path = str(log_path)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            record.command,
            cwd=REPO_ROOT,
            env=eval_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record.returncode = proc.returncode
    record.finished_at = time.time()
    return record


def start_process(record: RunRecord) -> tuple[subprocess.Popen, Any, RunRecord]:
    record.started_at = time.time()
    log_path = Path(record.output_dir) / f"{record.stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record.log_path = str(log_path)
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        record.command,
        cwd=REPO_ROOT,
        env=eval_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_file, record


def find_latest_h5(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.rglob("trajectories_batch_*.h5"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def find_videos(output_dir: Path) -> list[str]:
    return [str(path) for path in sorted(output_dir.rglob("*.mp4"))]


def validate_smoke(record: RunRecord, variant: Variant) -> RunRecord:
    if record.returncode != 0:
        record.smoke_ok = False
        record.smoke_reason = f"eval returned {record.returncode}"
        return record

    h5_path = find_latest_h5(Path(record.output_dir))
    if h5_path is None:
        record.smoke_ok = False
        record.smoke_reason = "no H5 trajectory produced"
        return record
    record.h5_path = str(h5_path)
    record.videos = find_videos(Path(record.output_dir))

    required_video_tokens = [
        "exo_camera_1",
        "head_camera",
        "wrist_camera_l",
        "wrist_camera_r",
    ]
    missing = [
        token for token in required_video_tokens if not any(token in video for video in record.videos)
    ]
    if missing:
        record.smoke_ok = False
        record.smoke_reason = f"missing videos: {missing}"
        return record

    try:
        with h5py.File(h5_path, "r") as f:
            traj = f["traj_0"]
            obj = traj["obs/extra/obj_start"][0][:3]
            requested = variant.object_pose[:3]
            pos_error = math.dist([float(v) for v in obj], requested)
            if float(obj[2]) < 0.70:
                record.smoke_ok = False
                record.smoke_reason = f"object z too low: {float(obj[2]):.3f}"
                return record
            if pos_error > 0.15:
                record.smoke_ok = False
                record.smoke_reason = f"object pose mismatch: {pos_error:.3f} m"
                return record
    except Exception as exc:
        record.smoke_ok = False
        record.smoke_reason = f"H5 validation failed: {exc}"
        return record

    record.smoke_ok = True
    record.smoke_reason = "ok"
    return record


def enrich_summary(h5_path: Path) -> dict[str, Any]:
    rows = summarize_h5(h5_path)
    if not rows:
        raise RuntimeError(f"No trajectories found in {h5_path}")
    row = rows[0]

    max_object_contacts = 0
    max_robot_object_contacts = 0
    left_finger_contact_any = False
    right_finger_contact_any = False
    contact_steps: list[int] = []
    with h5py.File(h5_path, "r") as f:
        traj = f[row["traj"]]
        n_steps = traj["obs/agent/qpos"].shape[0]
        for step in range(n_steps):
            info = task_info_at(traj, step)
            object_contacts = int(info.get("object_contact_count") or 0)
            robot_contacts = int(info.get("robot_object_contact_count") or 0)
            left_contact = bool(info.get("left_finger_contact"))
            right_contact = bool(info.get("right_finger_contact"))
            max_object_contacts = max(max_object_contacts, object_contacts)
            max_robot_object_contacts = max(max_robot_object_contacts, robot_contacts)
            left_finger_contact_any = left_finger_contact_any or left_contact
            right_finger_contact_any = right_finger_contact_any or right_contact
            if object_contacts or robot_contacts or left_contact or right_contact:
                contact_steps.append(step)

    row.update(
        {
            "max_object_contacts": max_object_contacts,
            "max_robot_object_contacts": max_robot_object_contacts,
            "left_finger_contact_any": left_finger_contact_any,
            "right_finger_contact_any": right_finger_contact_any,
            "contact_steps": contact_steps[:80],
        }
    )
    return row


def score_summary(summary: dict[str, Any]) -> float:
    score = 0.0
    obj_delta = float(summary.get("obj_delta_m") or 0.0)
    min_dist = float(summary.get("left_tcp_obj_dist_min_m") or summary["min_tcp_obj_dist_m"])
    if obj_delta > OBJECT_MOVE_THRESHOLD_M:
        score += 100000.0 + obj_delta * 100000.0
    if summary.get("left_held_any") or summary.get("right_held_any"):
        score += 50000.0
    if summary.get("left_touch_any") or summary.get("right_touch_any"):
        score += 20000.0
    if summary.get("left_finger_contact_any") or summary.get("right_finger_contact_any"):
        score += 5000.0
    score += float(summary.get("max_robot_object_contacts") or 0) * 500.0
    score += float(summary.get("max_object_contacts") or 0) * 50.0
    score += max(0.0, 1.0 - min_dist) * 1000.0
    return score


def is_promising(summary: dict[str, Any]) -> bool:
    obj_delta = float(summary.get("obj_delta_m") or 0.0)
    min_dist = float(summary.get("left_tcp_obj_dist_min_m") or summary["min_tcp_obj_dist_m"])
    has_contact = bool(
        summary.get("left_finger_contact_any")
        or summary.get("right_finger_contact_any")
        or (summary.get("max_robot_object_contacts") or 0) > 0
    )
    return (
        obj_delta > OBJECT_MOVE_THRESHOLD_M
        or summary.get("left_touch_any")
        or summary.get("right_touch_any")
        or summary.get("left_held_any")
        or summary.get("right_held_any")
        or min_dist < PROMISING_MIN_DIST_M
        or (has_contact and min_dist < PROMISING_CONTACT_MIN_DIST_M)
    )


def followup_depth(variant: Variant) -> int:
    return variant.name.count(CONTACT_Z_MARKER) + variant.name.count(TOWARD_TCP_MARKER)


def full_record_summary_by_name(records: list[RunRecord]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.stage == "full" and record.summary is not None:
            summaries[record.variant["name"]] = record.summary
    return summaries


def full_record_by_name(records: list[RunRecord]) -> dict[str, RunRecord]:
    return {
        record.variant["name"]: record
        for record in records
        if record.stage == "full" and record.summary is not None
    }


def summary_min_dist(summary: dict[str, Any]) -> float:
    return float(summary.get("left_tcp_obj_dist_min_m") or summary["min_tcp_obj_dist_m"])


def prune_pending(pending: list[Variant], records: list[RunRecord]) -> list[Variant]:
    summaries = full_record_summary_by_name(records)
    record_by_name = full_record_by_name(records)
    terminal_names = {
        record.variant["name"]
        for record in records
        if record.stage == "full" or (record.stage == "smoke" and record.smoke_ok is False)
    }
    pruned: list[Variant] = []
    seen: set[str] = set()
    for variant in pending:
        if variant.name in terminal_names:
            continue
        if variant.name in seen:
            continue
        seen.add(variant.name)

        parent_summary = summaries.get(variant.parent or "")
        is_contact_z = CONTACT_Z_MARKER in variant.name
        is_confirmation = variant.name.endswith("_default_confirm") or variant.name.endswith(
            "_contact_hold"
        )

        if followup_depth(variant) > 1:
            continue
        if parent_summary is not None and (is_contact_z or is_confirmation):
            parent_record = record_by_name.get(variant.parent or "")
            if is_contact_z and parent_record is not None:
                if parent_record.variant.get("mode") != "frozen":
                    continue
            parent_min_dist = summary_min_dist(parent_summary)
            if parent_min_dist > PROMISING_MIN_DIST_M:
                continue
            if variant.parent and CONTACT_Z_MARKER in variant.parent:
                continue

        pruned.append(variant)
    return pruned


def should_stop(summary: dict[str, Any]) -> bool:
    return float(summary.get("obj_delta_m") or 0.0) > OBJECT_MOVE_THRESHOLD_M


def make_run_record(
    variant: Variant,
    stage: str,
    benchmark_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    task_horizon: int,
) -> RunRecord:
    return RunRecord(
        variant=asdict(variant),
        stage=stage,
        output_dir=str(output_dir),
        benchmark_dir=str(benchmark_dir),
        command=eval_command(
            checkpoint_path=checkpoint_path,
            benchmark_dir=benchmark_dir,
            eval_config_cls=variant.eval_config_cls,
            task_horizon=task_horizon,
            output_dir=output_dir,
            num_workers=1,
        ),
    )


def report_path(output_root: Path, suffix: str) -> Path:
    return output_root / f"adaptive_report.{suffix}"


def write_reports(
    output_root: Path,
    records: list[RunRecord],
    pending: list[Variant],
    running: list[RunRecord],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "records": [asdict(record) for record in records],
        "running": [asdict(record) for record in running],
        "pending": [asdict(variant) for variant in pending],
    }
    report_path(output_root, "json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# RBY1 Adaptive Variant Search", ""]
    lines.append("## Completed")
    if records:
        for record in records:
            summary = record.summary or {}
            lines.append(
                "- "
                f"{record.variant['name']} [{record.variant['mode']}/{record.stage}] "
                f"rc={record.returncode} score={record.score} "
                f"move={summary.get('obj_delta_m')} "
                f"min_left_tcp={summary.get('left_tcp_obj_dist_min_m')} "
                f"contact={summary.get('left_finger_contact_any')} "
                f"success={summary.get('success_any')} "
                f"out={record.output_dir}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Running"])
    lines.extend(
        [
            f"- {record.variant['name']} [{record.variant['mode']}/{record.stage}]"
            for record in running
        ]
        or ["- none"]
    )
    lines.extend(["", "## Pending"])
    lines.extend(
        [
            f"- {variant.name} [{variant.mode}] priority={variant.priority}: {variant.reason}"
            for variant in pending
        ]
        or ["- none"]
    )
    report_path(output_root, "md").write_text("\n".join(lines) + "\n")


def add_followups(
    variant: Variant,
    summary: dict[str, Any],
    pending: list[Variant],
    seen_names: set[str],
) -> None:
    obj_delta = float(summary.get("obj_delta_m") or 0.0)
    min_dist = summary_min_dist(summary)
    has_contact = bool(
        summary.get("left_finger_contact_any")
        or summary.get("right_finger_contact_any")
        or (summary.get("max_robot_object_contacts") or 0) > 0
    )
    if obj_delta > OBJECT_MOVE_THRESHOLD_M:
        return

    if min_dist < 0.02 and not has_contact and TOWARD_TCP_MARKER not in variant.name:
        delta = summary.get("tcp_obj_delta_at_min_m") or [0.0, 0.0, 0.0]
        for scale in (0.5, 1.0):
            adjusted = list(variant.object_pose)
            adjusted[0] += max(-0.02, min(0.02, float(delta[0]) * scale))
            adjusted[1] += max(-0.02, min(0.02, float(delta[1]) * scale))
            name = f"{variant.name}_toward_tcp_{scale:g}"
            if name not in seen_names:
                seen_names.add(name)
                pending.append(
                    Variant(
                        name=name,
                        object_pose=adjusted,
                        robot_base_pose=list(variant.robot_base_pose),
                        scene_table_yaw_rad=variant.scene_table_yaw_rad,
                        priority=variant.priority + 30 + scale,
                        parent=variant.name,
                        reason="Added from close no-contact result.",
                    )
                )

    if (
        variant.mode == "frozen"
        and has_contact
        and obj_delta <= OBJECT_MOVE_THRESHOLD_M
        and min_dist <= PROMISING_MIN_DIST_M
        and CONTACT_Z_MARKER not in variant.name
    ):
        for dz in (0.01, -0.005):
            adjusted = list(variant.object_pose)
            adjusted[2] += dz
            name = f"{variant.name}_contact_z_{dz:+.3f}".replace("+", "plus").replace("-", "minus")
            if name not in seen_names:
                seen_names.add(name)
                pending.append(
                    Variant(
                        name=name,
                        object_pose=adjusted,
                        robot_base_pose=list(variant.robot_base_pose),
                        scene_table_yaw_rad=variant.scene_table_yaw_rad,
                        priority=variant.priority + 20,
                        parent=variant.name,
                        reason="Added from contact without movement.",
                    )
                )


def variant_run_dirs(output_root: Path, variant: Variant, stage: str) -> tuple[Path, Path]:
    safe_name = variant.name.replace("/", "_")
    mode_dir = output_root / safe_name / variant.mode
    return mode_dir / "benchmark", mode_dir / stage


def run_smoke(
    variant: Variant,
    output_root: Path,
    checkpoint_path: Path,
    smoke_horizon: int,
) -> RunRecord:
    benchmark_dir, output_dir = variant_run_dirs(output_root, variant, "smoke")
    create_benchmark(variant, benchmark_dir)
    record = make_run_record(
        variant=variant,
        stage="smoke",
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        task_horizon=smoke_horizon,
    )
    record = run_blocking(record)
    return validate_smoke(record, variant)


def prepare_full_record(
    variant: Variant,
    output_root: Path,
    checkpoint_path: Path,
    task_horizon: int,
) -> RunRecord:
    benchmark_dir, output_dir = variant_run_dirs(output_root, variant, "full")
    create_benchmark(variant, benchmark_dir)
    return make_run_record(
        variant=variant,
        stage="full",
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        task_horizon=task_horizon,
    )


def finalize_full_record(record: RunRecord) -> RunRecord:
    h5_path = find_latest_h5(Path(record.output_dir))
    if h5_path is not None:
        record.h5_path = str(h5_path)
        record.videos = find_videos(Path(record.output_dir))
        record.summary = enrich_summary(h5_path)
        record.score = score_summary(record.summary)
    else:
        record.score = -100000.0
    return record


def confirmation_variant(variant: Variant) -> Variant:
    confirmed = copy.deepcopy(variant)
    confirmed.name = f"{variant.name}_default_confirm"
    confirmed.mode = "default"
    confirmed.eval_config_cls = DEFAULT_CONFIG
    confirmed.priority = variant.priority - 0.1
    confirmed.parent = variant.name
    confirmed.reason = "Default/base-active confirmation of promising frozen result."
    return confirmed


def contact_hold_variant(variant: Variant) -> Variant:
    adjusted = copy.deepcopy(variant)
    adjusted.name = f"{variant.name}_contact_hold"
    adjusted.mode = "contact_hold"
    adjusted.eval_config_cls = CONTACT_HOLD_CONFIG
    adjusted.priority = variant.priority - 0.2
    adjusted.parent = variant.name
    adjusted.reason = "Contact-hold confirmation after useful geometry contact."
    return adjusted


def load_resume_state(output_root: Path) -> tuple[list[RunRecord], list[Variant]] | None:
    path = report_path(output_root, "json")
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    completed = [RunRecord(**record) for record in payload.get("records", [])]
    pending = [Variant(**variant) for variant in payload.get("pending", [])]
    pending.extend(Variant(**record["variant"]) for record in payload.get("running", []))
    full_names = {record.variant["name"] for record in completed if record.stage == "full"}
    pending_names = {variant.name for variant in pending}
    for record in completed:
        name = record.variant["name"]
        if record.stage == "smoke" and record.smoke_ok and name not in full_names:
            if name not in pending_names:
                pending.append(Variant(**record.variant))
                pending_names.add(name)
    return completed, pending


def run_search(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_path)
    completed: list[RunRecord] = []
    resume_state = load_resume_state(output_root) if args.resume else None
    if resume_state is None:
        pending = initial_variants()
    else:
        completed, pending = resume_state
        pending = prune_pending(pending, completed)
        print(
            f"[resume] loaded completed={len(completed)} pending={len(pending)}",
            flush=True,
        )
        write_reports(output_root, completed, pending, [])
    if args.limit is not None:
        pending = pending[: args.limit]
    seen_names = {variant.name for variant in pending}
    seen_names.update(record.variant["name"] for record in completed)
    running: list[tuple[subprocess.Popen, Any, RunRecord]] = []
    stop_requested = False
    max_parallel = args.max_parallel

    if args.dry_run:
        for variant in sorted(pending, key=lambda item: item.priority, reverse=True):
            benchmark_dir, _ = variant_run_dirs(output_root, variant, "dry_run")
            create_benchmark(variant, benchmark_dir)
            print(f"{variant.name}: benchmark={benchmark_dir}")
        return

    while pending or running:
        pending.sort(key=lambda item: item.priority, reverse=True)
        while pending and len(running) < max_parallel and not stop_requested:
            variant = pending.pop(0)
            print(f"[smoke] {variant.name} ({variant.mode})", flush=True)
            smoke_record = run_smoke(
                variant=variant,
                output_root=output_root,
                checkpoint_path=checkpoint_path,
                smoke_horizon=args.smoke_horizon,
            )
            completed.append(smoke_record)
            if not smoke_record.smoke_ok:
                print(f"[smoke failed] {variant.name}: {smoke_record.smoke_reason}", flush=True)
                write_reports(output_root, completed, pending, [item[2] for item in running])
                continue

            full_record = prepare_full_record(
                variant=variant,
                output_root=output_root,
                checkpoint_path=checkpoint_path,
                task_horizon=args.task_horizon,
            )
            print(f"[full start] {variant.name} ({variant.mode})", flush=True)
            running.append(start_process(full_record))
            write_reports(output_root, completed, pending, [item[2] for item in running])

        if not running:
            break

        time.sleep(args.poll_seconds)
        still_running: list[tuple[subprocess.Popen, Any, RunRecord]] = []
        for proc, log_file, record in running:
            returncode = proc.poll()
            if returncode is None:
                still_running.append((proc, log_file, record))
                continue

            log_file.close()
            record.returncode = returncode
            record.finished_at = time.time()
            record = finalize_full_record(record)
            completed.append(record)
            summary = record.summary or {}
            print(
                f"[full done] {record.variant['name']} rc={returncode} "
                f"score={record.score} move={summary.get('obj_delta_m')} "
                f"min_left_tcp={summary.get('left_tcp_obj_dist_min_m')}",
                flush=True,
            )

            if returncode != 0 and record.log_path:
                log_text = Path(record.log_path).read_text(errors="replace")
                if "out of memory" in log_text.lower() and max_parallel > 1:
                    max_parallel = 1
                    print("[parallel fallback] CUDA OOM detected; max_parallel=1", flush=True)

            variant = Variant(**record.variant)
            if record.summary is not None:
                if should_stop(record.summary):
                    stop_requested = True
                    pending.clear()
                    print(f"[stop] object moved in {variant.name}", flush=True)
                else:
                    add_followups(variant, record.summary, pending, seen_names)
                    pending = prune_pending(pending, completed)
                    if (
                        variant.mode == "frozen"
                        and is_promising(record.summary)
                        and followup_depth(variant) == 0
                        and not args.skip_default_confirm
                    ):
                        confirm = confirmation_variant(variant)
                        if confirm.name not in seen_names:
                            seen_names.add(confirm.name)
                            pending.append(confirm)
                    if (
                        variant.mode == "frozen"
                        and bool(record.summary.get("left_finger_contact_any"))
                        and followup_depth(variant) == 0
                    ):
                        contact_hold = contact_hold_variant(variant)
                        if contact_hold.name not in seen_names:
                            seen_names.add(contact_hold.name)
                            pending.append(contact_hold)
                    pending = prune_pending(pending, completed)

            write_reports(output_root, completed, pending, [item[2] for item in still_running])

        running = still_running

    write_reports(output_root, completed, pending, [item[2] for item in running])
    print(f"Report: {report_path(output_root, 'md')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adaptive RBY1 salt-shaker geometry variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--output_root",
        default="eval_output/rby1_adaptive_geometry_search",
        help="Root directory for benchmarks, eval outputs, and reports.",
    )
    parser.add_argument("--task_horizon", type=int, default=300)
    parser.add_argument("--smoke_horizon", type=int, default=25)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit initial variants.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from adaptive_report.json.")
    parser.add_argument("--skip_default_confirm", action="store_true")
    args = parser.parse_args()

    run_search(args)


if __name__ == "__main__":
    main()
