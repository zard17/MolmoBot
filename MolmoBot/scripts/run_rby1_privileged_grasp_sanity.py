#!/usr/bin/env python3
"""Run a CuRobo-free scripted RBY1 grasp sanity check on the salt-shaker scene."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rby1_adaptive_variants import (
    DEFAULT_CHECKPOINT,
    RunRecord,
    Variant,
    create_benchmark,
    find_latest_h5,
    find_videos,
    make_run_record,
    run_blocking,
    validate_smoke,
)
from scripts.summarize_rby1_eval import summarize_h5


SCRIPTED_CONFIG = (
    "olmo.eval.configure_molmo_spaces:MolmoBotRBY1ScriptedGraspSanityEvalConfig"
)
BEST_OBJECT_POSE = [0.377, 0.278, 0.807, 0.7071068, 0.7071068, 0.0, 0.0]
OUTPUT_ROOT = REPO_ROOT / "eval_output" / "rby1_scripted_grasp_sweep_ctrl100_local_arm"
DEFAULT_SOURCE_H5 = (
    REPO_ROOT
    / "eval_output/rby1_midpoint_sweep/mid_object_x_minus_075_y_plus_010/frozen/full/"
    / "MolmoBotRBY1PickPnPFrozenBaseEvalConfig/20260415_121229/house_0/"
    / "trajectories_batch_1_of_1.h5"
)


def report_path(output_root: Path, suffix: str) -> Path:
    return output_root / f"scripted_grasp_sanity_report.{suffix}"


def load_records(output_root: Path) -> list[RunRecord]:
    path = report_path(output_root, "json")
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [RunRecord(**record) for record in payload.get("records", [])]


def enrich_summary(h5_path: Path) -> dict[str, Any]:
    rows = summarize_h5(h5_path)
    if not rows:
        raise RuntimeError(f"No trajectories found in {h5_path}")
    row = rows[0]

    with h5py.File(h5_path, "r") as f:
        traj = f[row["traj"]]
        obj = np.asarray(traj["obs/extra/obj_start"])[:, :3]

    obj_delta = float(row.get("obj_delta_m") or 0.0)
    z_delta = float(obj[-1, 2] - obj[0, 2])
    z_lift_max = float(np.max(obj[:, 2]) - obj[0, 2])
    left_mid = row.get("left_finger_midpoint_obj_dist_min_m")
    right_mid = row.get("right_finger_midpoint_obj_dist_min_m")
    row["obj_z_delta_m"] = z_delta
    row["obj_z_lift_max_m"] = z_lift_max
    row["physical_grasp_success"] = bool(
        z_lift_max > 0.03
        or obj_delta > 0.03
        or row.get("left_held_any")
        or row.get("right_held_any")
    )
    row["diagnostic"] = {
        "obj_delta_m": obj_delta,
        "obj_z_delta_m": z_delta,
        "obj_z_lift_max_m": z_lift_max,
        "left_midpoint_min_m": left_mid,
        "right_midpoint_min_m": right_mid,
        "success_rule": "z_lift_max>0.03 or obj_delta>0.03 or held_any",
    }
    return row


def score_summary(summary: dict[str, Any] | None) -> float:
    if not summary:
        return -1.0
    score = 0.0
    z_lift = float(summary.get("obj_z_lift_max_m") or 0.0)
    obj_delta = float(summary.get("obj_delta_m") or 0.0)
    left_mid = summary.get("left_finger_midpoint_obj_dist_min_m")
    if summary.get("physical_grasp_success"):
        score += 1_000_000.0
    if summary.get("left_held_any") or summary.get("right_held_any"):
        score += 100_000.0
    if summary.get("left_touch_any") or summary.get("right_touch_any"):
        score += 50_000.0
    score += z_lift * 10_000.0 + obj_delta * 1_000.0
    if left_mid is not None:
        score -= float(left_mid) * 100.0
    return score


def decode_json_bytes(value) -> dict[str, Any] | None:
    raw = bytes(value.tolist()).split(b"\x00", 1)[0]
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def qpos_at(traj, step: int) -> dict[str, list[float]]:
    qpos = decode_json_bytes(traj["obs/agent/qpos"][step])
    if qpos is None:
        raise RuntimeError(f"No qpos found at source step {step}")
    return qpos


def copy_move_groups(qpos: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        "base": list(qpos["base"]),
        "torso": list(qpos["torso"]),
        "left_arm": list(qpos["left_arm"]),
    }


def apply_offsets(
    targets: dict[str, dict[str, list[float]]],
    *,
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    left_arm_offset: dict[int, float] | None = None,
    torso_offset: dict[int, float] | None = None,
    phases: tuple[str, ...] = ("approach", "grasp", "lift"),
) -> dict[str, dict[str, list[float]]]:
    adjusted = json.loads(json.dumps(targets))
    for phase, target in adjusted.items():
        if phase not in phases:
            continue
        if "base" in target:
            for idx, delta in enumerate(base_offset):
                target["base"][idx] += delta
        if left_arm_offset and "left_arm" in target:
            for idx, delta in left_arm_offset.items():
                target["left_arm"][idx] += delta
        if torso_offset and "torso" in target:
            for idx, delta in torso_offset.items():
                target["torso"][idx] += delta
    return adjusted


def load_source_targets(source_h5: Path) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    rows = summarize_h5(source_h5)
    if not rows:
        raise RuntimeError(f"No trajectories found in source H5: {source_h5}")
    row = rows[0]
    grasp_step = int(
        row.get("left_finger_midpoint_obj_dist_min_step")
        or row.get("left_tcp_obj_dist_min_step")
        or row["min_dist_step"]
    )
    with h5py.File(source_h5, "r") as f:
        traj = f[row["traj"]]
        n_steps = traj["obs/agent/qpos"].shape[0]
        approach_step = max(0, grasp_step - 45)
        lift_step = min(n_steps - 1, grasp_step + 55)
        targets = {
            "approach": copy_move_groups(qpos_at(traj, approach_step)),
            "grasp": copy_move_groups(qpos_at(traj, grasp_step)),
            "lift": copy_move_groups(qpos_at(traj, lift_step)),
        }
    metadata = {
        "source_h5": str(source_h5),
        "source_traj": row["traj"],
        "approach_step": approach_step,
        "grasp_step": grasp_step,
        "lift_step": lift_step,
        "source_left_midpoint_min_m": row.get("left_finger_midpoint_obj_dist_min_m"),
    }
    return targets, metadata


def scripted_config(
    targets: dict[str, dict[str, list[float]]],
    phase_steps: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "selected_arm": "left",
        "targets": targets,
        "phase_steps": phase_steps
        or {
            "open": 8,
            "approach": 32,
            "grasp": 24,
            "close": 35,
            "lift": 100,
            "hold": 40,
        },
        "open_gripper_command": -100.0,
        "close_gripper_command": 100.0,
    }


def build_scripted_variants(
    base_targets: dict[str, dict[str, list[float]]]
) -> list[tuple[Variant, dict[str, Any]]]:
    close_early_hold = {
        "open": 8,
        "approach": 28,
        "grasp": 16,
        "close": 50,
        "lift": 110,
        "hold": 50,
    }
    candidates = [
        (
            "baseline_replay",
            apply_offsets(base_targets),
            "Replay closest-contact qpos from best frozen run with earlier closure.",
        ),
        (
            "torso_pitch_plus_030",
            apply_offsets(base_targets, torso_offset={1: 0.03}),
            "Previous best smoke signal: small torso pitch offset.",
        ),
        (
            "torso_pitch_plus_060",
            apply_offsets(base_targets, torso_offset={1: 0.06}),
            "Increase torso bend to bring the gripper closer to the table.",
        ),
        (
            "torso_pitch_plus_030_base_y_minus_020",
            apply_offsets(
                base_targets,
                base_offset=(0.0, -0.02, 0.0),
                torso_offset={1: 0.03},
            ),
            "Combine previous best torso bend with the best base-y direction.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_020",
            apply_offsets(
                base_targets,
                base_offset=(-0.02, 0.0, 0.0),
                torso_offset={1: 0.03},
            ),
            "Combine torso bend with the better base-x direction from smoke.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_040",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                torso_offset={1: 0.03},
            ),
            "Continue the helpful base-x shift to test whether the near miss is still short.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_060",
            apply_offsets(
                base_targets,
                base_offset=(-0.06, 0.0, 0.0),
                torso_offset={1: 0.03},
            ),
            "Bracket the base-x correction with a larger shift.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_020_y_plus_010",
            apply_offsets(
                base_targets,
                base_offset=(-0.02, 0.01, 0.0),
                torso_offset={1: 0.03},
            ),
            "Add a small positive y correction around the best base-x shift.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_020_y_minus_010",
            apply_offsets(
                base_targets,
                base_offset=(-0.02, -0.01, 0.0),
                torso_offset={1: 0.03},
            ),
            "Add a smaller negative y correction than the failed -2cm sweep.",
        ),
        (
            "torso_pitch_plus_030_base_x_minus_040_y_plus_010",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.01, 0.0),
                torso_offset={1: 0.03},
            ),
            "Combine the stronger base-x shift with a small positive y correction.",
        ),
        (
            "torso_pitch_plus_020_base_x_minus_040",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                torso_offset={1: 0.02},
            ),
            "Test whether slightly less torso bend improves fingertip alignment.",
        ),
        (
            "torso_pitch_plus_040_base_x_minus_040",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                torso_offset={1: 0.04},
            ),
            "Test whether slightly more torso bend improves fingertip alignment.",
        ),
        (
            "base_x_minus_040_wrist_roll_plus_060",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={6: 0.06},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Try a smaller wrist roll correction at the refined base position.",
        ),
        (
            "base_x_minus_040_wrist_roll_minus_060",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={6: -0.06},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Try the opposite smaller wrist roll correction at the refined base position.",
        ),
        (
            "local_j0_plus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={0: 0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 0 positive, targeting the measured lateral miss.",
        ),
        (
            "local_j0_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={0: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 0 negative, bracketing lateral miss direction.",
        ),
        (
            "local_j1_plus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={1: 0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 1 positive, testing vertical/lateral coupling.",
        ),
        (
            "local_j1_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={1: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 1 negative, testing vertical/lateral coupling.",
        ),
        (
            "local_j2_plus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={2: 0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 2 positive, testing wrist line-up before closure.",
        ),
        (
            "local_j2_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={2: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 2 negative, testing wrist line-up before closure.",
        ),
        (
            "local_j5_plus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={5: 0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 5 positive, testing distal vertical alignment.",
        ),
        (
            "local_j5_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={5: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Local grasp correction: joint 5 negative, testing distal vertical alignment.",
        ),
        (
            "local_j0_plus_080_j5_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={0: 0.08, 5: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Combine the likely lateral and distal-down corrections if both are helpful.",
        ),
        (
            "local_j0_minus_080_j5_minus_080",
            apply_offsets(
                base_targets,
                base_offset=(-0.04, 0.0, 0.0),
                left_arm_offset={0: -0.08, 5: -0.08},
                torso_offset={1: 0.04},
                phases=("grasp", "lift"),
            ),
            "Alternate combined correction for the measured lateral/upward miss.",
        ),
        (
            "grasp_elbow_lower_plus_080",
            apply_offsets(
                base_targets,
                left_arm_offset={3: 0.08},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Perturb the grasp/lift elbow joint while keeping approach stable.",
        ),
        (
            "grasp_elbow_lower_minus_080",
            apply_offsets(
                base_targets,
                left_arm_offset={3: -0.08},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Opposite elbow perturbation to bracket the vertical near miss.",
        ),
        (
            "wrist_roll_plus_120_torso_pitch_plus_030",
            apply_offsets(
                base_targets,
                left_arm_offset={6: 0.12},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Increase wrist roll during closure to test enclosure geometry.",
        ),
        (
            "wrist_roll_minus_120_torso_pitch_plus_030",
            apply_offsets(
                base_targets,
                left_arm_offset={6: -0.12},
                torso_offset={1: 0.03},
                phases=("grasp", "lift"),
            ),
            "Opposite wrist roll during closure to test enclosure geometry.",
        ),
    ]

    variants: list[tuple[Variant, dict[str, Any]]] = []
    for name, targets, reason in candidates:
        variants.append(
            (
                Variant(
                    name=name,
                    object_pose=list(BEST_OBJECT_POSE),
                    eval_config_cls=SCRIPTED_CONFIG,
                    mode="scripted",
                    reason=reason,
                ),
                scripted_config(targets, close_early_hold),
            )
        )
    return variants


def write_report(output_root: Path, records: list[RunRecord]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "config": SCRIPTED_CONFIG,
        "object_pose": BEST_OBJECT_POSE,
        "records": [asdict(record) for record in records],
    }
    report_path(output_root, "json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# RBY1 Scripted Grasp Sanity Check",
        "",
        "This run bypasses the learned policy and CuRobo. It replays fixed RBY1 joint",
        "waypoints to test whether the current salt-shaker scene is physically pickable.",
        "",
        f"- object pose: `{BEST_OBJECT_POSE}`",
        f"- eval config: `{SCRIPTED_CONFIG}`",
        "",
        "## Runs",
    ]
    if not records:
        lines.append("- none")
    for record in records:
        summary = record.summary or {}
        diagnostic = summary.get("diagnostic") or {}
        lines.append(
            "- "
            f"{record.stage}: rc={record.returncode} "
            f"smoke_ok={record.smoke_ok} "
            f"physical_grasp_success={summary.get('physical_grasp_success')} "
            f"obj_delta={diagnostic.get('obj_delta_m')} "
            f"z_delta={diagnostic.get('obj_z_delta_m')} "
            f"z_lift_max={diagnostic.get('obj_z_lift_max_m')} "
            f"success_any={summary.get('success_any')} "
            f"score={record.score} "
            f"h5={record.h5_path}"
        )
    report_path(output_root, "md").write_text("\n".join(lines) + "\n")


def record_variant_name(record: RunRecord) -> str:
    variant = record.variant
    if isinstance(variant, dict):
        return str(variant.get("name"))
    return str(variant.name)


def completed_variant_stage(records: list[RunRecord], variant: Variant, stage: str) -> bool:
    return any(
        record_variant_name(record) == variant.name
        and record.stage == stage
        and record.returncode == 0
        for record in records
    )


def run_stage(
    variant: Variant,
    stage: str,
    output_root: Path,
    checkpoint_path: Path,
    task_horizon: int,
    policy_payload: dict[str, Any],
) -> RunRecord:
    benchmark_dir = output_root / variant.name / variant.mode / "benchmark"
    create_benchmark(variant, benchmark_dir)
    output_dir = output_root / variant.name / variant.mode / stage
    record = make_run_record(
        variant,
        stage,
        benchmark_dir,
        output_dir,
        checkpoint_path,
        task_horizon,
    )
    record.command = [
        "env",
        f"RBY1_SCRIPTED_GRASP_CONFIG_JSON={json.dumps(policy_payload)}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source_h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--smoke_horizon", type=int, default=40)
    parser.add_argument("--task_horizon", type=int, default=400)
    parser.add_argument(
        "--variant",
        action="append",
        help="Run only the named variant. Can be passed multiple times.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_targets, source_metadata = load_source_targets(args.source_h5)
    variants = build_scripted_variants(base_targets)
    if args.variant:
        requested = set(args.variant)
        variants = [(variant, payload) for variant, payload in variants if variant.name in requested]
        missing = requested - {variant.name for variant, _ in variants}
        if missing:
            raise ValueError(f"Unknown variant(s): {sorted(missing)}")

    records = load_records(args.output_root) if args.resume else []

    for variant, payload in variants:
        if completed_variant_stage(records, variant, "smoke"):
            continue
        smoke_record = run_stage(
            variant,
            "smoke",
            args.output_root,
            args.checkpoint_path,
            args.smoke_horizon,
            payload,
        )
        smoke_record = validate_smoke(smoke_record, variant)
        if smoke_record.summary is None and smoke_record.h5_path:
            smoke_record.summary = enrich_summary(Path(smoke_record.h5_path))
            smoke_record.score = score_summary(smoke_record.summary)
        records.append(smoke_record)
        write_report(args.output_root, records)
        print(
            f"[smoke] {variant.name} ok={smoke_record.smoke_ok} "
            f"score={smoke_record.score} reason={smoke_record.smoke_reason}"
        )
        if smoke_record.summary and smoke_record.summary.get("physical_grasp_success"):
            break

    smoke_candidates = [
        record
        for record in records
        if record.stage == "smoke" and record.returncode == 0 and record.h5_path
    ]
    if not smoke_candidates:
        print("[smoke failed] no scripted smoke produced an H5 trajectory")
        return

    best_smoke = max(smoke_candidates, key=lambda record: record.score or -1.0)
    best_variant_pair = next(
        (variant, payload)
        for variant, payload in variants
        if variant.name == record_variant_name(best_smoke)
    )
    best_variant, best_payload = best_variant_pair

    if not completed_variant_stage(records, best_variant, "full"):
        full_record = run_stage(
            best_variant,
            "full",
            args.output_root,
            args.checkpoint_path,
            args.task_horizon,
            best_payload,
        )
        records.append(full_record)
        write_report(args.output_root, records)
        summary = full_record.summary or {}
        print(
            "[full done] "
            f"physical_grasp_success={summary.get('physical_grasp_success')} "
            f"obj_delta={summary.get('obj_delta_m')} "
            f"z_delta={summary.get('obj_z_delta_m')} "
            f"z_lift_max={summary.get('obj_z_lift_max_m')} "
            f"h5={full_record.h5_path}"
        )
    else:
        write_report(args.output_root, records)
        print("[resume] full stage already complete")

    print(f"Source: {source_metadata}")
    print(f"Report: {report_path(args.output_root, 'md')}")


if __name__ == "__main__":
    main()
