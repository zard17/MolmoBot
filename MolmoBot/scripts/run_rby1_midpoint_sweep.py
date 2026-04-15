#!/usr/bin/env python3
"""Run RBY1 Pick/PnP variants ranked by finger-midpoint distance.

This is a narrower follow-up to the adaptive geometry search. The earlier
search found several variants where the RBY1 TCP/site comes very close to the
salt shaker while the actual left-finger pinch center remains about 10 cm away.
This runner therefore ranks by the left finger midpoint, not the TCP.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_rby1_adaptive_variants import (
    BASE_OBJECT_POSE,
    CONTACT_HOLD_CONFIG,
    DEFAULT_CHECKPOINT,
    FROZEN_CONFIG,
    ROBOT_BASE_POSE,
    RunRecord,
    Variant,
    create_benchmark,
    find_latest_h5,
    find_videos,
    make_run_record,
    report_path as adaptive_report_path,
    run_blocking,
    validate_smoke,
    with_robot_yaw,
)
from scripts.summarize_rby1_eval import summarize_h5, task_info_at


OBJECT_MOVE_THRESHOLD_M = 0.005


def shifted_pose(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> list[float]:
    pose = list(BASE_OBJECT_POSE)
    pose[0] += dx
    pose[1] += dy
    pose[2] += dz
    return pose


def midpoint_variants() -> list[Variant]:
    base = list(BASE_OBJECT_POSE)
    return [
        Variant(
            "mid_baseline_reference",
            base,
            priority=100,
            reason="Reference using current best benchmark pose.",
        ),
        Variant(
            "mid_table_yaw_plus_5deg",
            base,
            scene_table_yaw_rad=math.radians(5.0),
            priority=95,
            reason="Previously best finger-midpoint variant.",
        ),
        Variant(
            "mid_table_yaw_plus_7p5deg",
            base,
            scene_table_yaw_rad=math.radians(7.5),
            priority=94,
            reason="Refine around table +5 deg.",
        ),
        Variant(
            "mid_table_yaw_plus_10deg",
            base,
            scene_table_yaw_rad=math.radians(10.0),
            priority=93,
            reason="Refine around table +5 deg.",
        ),
        Variant(
            "mid_object_z_plus_030",
            shifted_pose(dz=0.030),
            priority=90,
            reason="Move object toward observed higher finger midpoint.",
        ),
        Variant(
            "mid_object_x_minus_030",
            shifted_pose(dx=-0.030),
            priority=89,
            reason="Move object toward observed finger midpoint behind TCP.",
        ),
        Variant(
            "mid_object_x_minus_045",
            shifted_pose(dx=-0.045),
            priority=88,
            reason="Stronger x shift toward finger midpoint.",
        ),
        Variant(
            "mid_object_x_minus_060",
            shifted_pose(dx=-0.060),
            priority=87.5,
            reason="Check whether the x correction keeps improving beyond 4.5 cm.",
        ),
        Variant(
            "mid_object_x_minus_075",
            shifted_pose(dx=-0.075),
            priority=87.25,
            reason="Check whether the x correction keeps improving beyond 6 cm.",
        ),
        Variant(
            "mid_object_x_minus_090",
            shifted_pose(dx=-0.090),
            priority=87.125,
            reason="Check whether the x correction keeps improving beyond 7.5 cm.",
        ),
        Variant(
            "mid_object_x_minus_075_z_plus_015",
            shifted_pose(dx=-0.075, dz=0.015),
            priority=87.05,
            reason="Refine current best with small positive object height.",
        ),
        Variant(
            "mid_object_x_minus_075_z_plus_030",
            shifted_pose(dx=-0.075, dz=0.030),
            priority=87.04,
            reason="Refine current best with larger positive object height.",
        ),
        Variant(
            "mid_object_x_minus_075_y_plus_010",
            shifted_pose(dx=-0.075, dy=0.010),
            priority=87.03,
            reason="Small lateral refinement around current best.",
        ),
        Variant(
            "mid_object_x_minus_075_y_minus_010",
            shifted_pose(dx=-0.075, dy=-0.010),
            priority=87.02,
            reason="Small lateral refinement around current best.",
        ),
        Variant(
            "mid_object_x_minus_075_table_yaw_plus_7p5deg",
            shifted_pose(dx=-0.075),
            scene_table_yaw_rad=math.radians(7.5),
            priority=87.01,
            reason="Combine current best x shift with best table yaw.",
        ),
        Variant(
            "mid_object_x_minus_075_contact_hold",
            shifted_pose(dx=-0.075),
            eval_config_cls=CONTACT_HOLD_CONFIG,
            mode="contact_hold",
            priority=87.0,
            reason="Contact-hold confirmation on current best x-shift geometry.",
        ),
        Variant(
            "mid_object_x_minus_030_z_plus_030",
            shifted_pose(dx=-0.030, dz=0.030),
            priority=87,
            reason="Move object toward finger midpoint in x and z.",
        ),
        Variant(
            "mid_object_x_minus_045_z_plus_045",
            shifted_pose(dx=-0.045, dz=0.045),
            priority=86,
            reason="Stronger combined x/z shift toward finger midpoint.",
        ),
        Variant(
            "mid_table_yaw_plus_5deg_x_minus_030",
            shifted_pose(dx=-0.030),
            scene_table_yaw_rad=math.radians(5.0),
            priority=85,
            reason="Combine best table yaw with x midpoint correction.",
        ),
        Variant(
            "mid_table_yaw_plus_5deg_z_plus_030",
            shifted_pose(dz=0.030),
            scene_table_yaw_rad=math.radians(5.0),
            priority=84,
            reason="Combine best table yaw with z midpoint correction.",
        ),
        Variant(
            "mid_table_yaw_plus_5deg_x_minus_030_z_plus_030",
            shifted_pose(dx=-0.030, dz=0.030),
            scene_table_yaw_rad=math.radians(5.0),
            priority=83,
            reason="Combine table yaw with x/z midpoint correction.",
        ),
        Variant(
            "mid_robot_yaw_plus_10deg_table_yaw_plus_5deg",
            base,
            robot_base_pose=with_robot_yaw(10.0),
            scene_table_yaw_rad=math.radians(5.0),
            priority=82,
            reason="Combine previous good robot yaw with table yaw.",
        ),
    ]


def report_path(output_root: Path, suffix: str) -> Path:
    return output_root / f"midpoint_sweep_report.{suffix}"


def load_records(output_root: Path) -> list[RunRecord]:
    path = report_path(output_root, "json")
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [RunRecord(**record) for record in payload.get("records", [])]


def enrich_midpoint_summary(h5_path: Path) -> dict[str, Any]:
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


def midpoint_distance(summary: dict[str, Any]) -> float:
    value = summary.get("left_finger_midpoint_obj_dist_min_m")
    if value is None:
        return float("inf")
    return float(value)


def score_summary(summary: dict[str, Any]) -> float:
    obj_delta = float(summary.get("obj_delta_m") or 0.0)
    mid_dist = midpoint_distance(summary)
    score = 0.0
    if obj_delta > OBJECT_MOVE_THRESHOLD_M:
        score += 100000.0 + obj_delta * 100000.0
    if summary.get("left_held_any") or summary.get("right_held_any"):
        score += 50000.0
    if summary.get("left_touch_any") or summary.get("right_touch_any"):
        score += 20000.0
    if summary.get("left_finger_contact_any") or summary.get("right_finger_contact_any"):
        score += 5000.0
    score += float(summary.get("max_robot_object_contacts") or 0) * 100.0
    score += float(summary.get("max_object_contacts") or 0) * 5.0
    if math.isfinite(mid_dist):
        score += max(0.0, 1.0 - mid_dist) * 100000.0
    return score


def finalize_record(record: RunRecord) -> RunRecord:
    h5_path = find_latest_h5(Path(record.output_dir))
    if h5_path is None:
        record.score = -100000.0
        return record
    record.h5_path = str(h5_path)
    record.videos = find_videos(Path(record.output_dir))
    record.summary = enrich_midpoint_summary(h5_path)
    record.score = score_summary(record.summary)
    return record


def completed_names(records: list[RunRecord]) -> set[str]:
    return {
        record.variant["name"]
        for record in records
        if record.stage == "full" or (record.stage == "smoke" and record.smoke_ok is False)
    }


def write_report(output_root: Path, records: list[RunRecord], pending: list[Variant]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path(output_root, "json").write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "records": [asdict(record) for record in records],
                "pending": [asdict(variant) for variant in pending],
            },
            indent=2,
        )
        + "\n"
    )

    lines = ["# RBY1 Midpoint Sweep", ""]
    lines.append("## Completed")
    if not records:
        lines.append("- none")
    for record in records:
        summary = record.summary or {}
        lines.append(
            "- "
            f"{record.variant['name']} [{record.stage}] rc={record.returncode} "
            f"score={record.score} move={summary.get('obj_delta_m')} "
            f"left_mid_min={summary.get('left_finger_midpoint_obj_dist_min_m')} "
            f"left_mid_step={summary.get('left_finger_midpoint_obj_dist_min_step')} "
            f"left_tcp_min={summary.get('left_tcp_obj_dist_min_m')} "
            f"contact={summary.get('left_finger_contact_any')} "
            f"robot_contacts={summary.get('max_robot_object_contacts')} "
            f"success={summary.get('success_any')} "
            f"h5={record.h5_path}"
        )

    lines.extend(["", "## Pending"])
    if not pending:
        lines.append("- none")
    for variant in pending:
        lines.append(f"- {variant.name} priority={variant.priority}: {variant.reason}")
    report_path(output_root, "md").write_text("\n".join(lines) + "\n")


def variant_dirs(output_root: Path, variant: Variant, stage: str) -> tuple[Path, Path]:
    root = output_root / variant.name / variant.mode
    return root / "benchmark", root / stage


def run_variant(
    variant: Variant,
    output_root: Path,
    checkpoint_path: Path,
    smoke_horizon: int,
    task_horizon: int,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    benchmark_dir, smoke_dir = variant_dirs(output_root, variant, "smoke")
    create_benchmark(variant, benchmark_dir)
    smoke = make_run_record(
        variant=variant,
        stage="smoke",
        benchmark_dir=benchmark_dir,
        output_dir=smoke_dir,
        checkpoint_path=checkpoint_path,
        task_horizon=smoke_horizon,
    )
    smoke = validate_smoke(run_blocking(smoke), variant)
    records.append(smoke)
    if not smoke.smoke_ok:
        return records

    benchmark_dir, full_dir = variant_dirs(output_root, variant, "full")
    create_benchmark(variant, benchmark_dir)
    full = make_run_record(
        variant=variant,
        stage="full",
        benchmark_dir=benchmark_dir,
        output_dir=full_dir,
        checkpoint_path=checkpoint_path,
        task_horizon=task_horizon,
    )
    full = finalize_record(run_blocking(full))
    records.append(full)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_root", default="eval_output/rby1_midpoint_sweep")
    parser.add_argument("--task_horizon", type=int, default=400)
    parser.add_argument("--smoke_horizon", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_path)
    records = load_records(output_root) if args.resume else []
    done = completed_names(records)

    variants = sorted(midpoint_variants(), key=lambda item: item.priority, reverse=True)
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"Unknown variants: {sorted(missing)}")
    variants = [variant for variant in variants if variant.name not in done]
    if args.limit is not None:
        variants = variants[: args.limit]

    if args.dry_run:
        for variant in variants:
            benchmark_dir, _ = variant_dirs(output_root, variant, "dry_run")
            test_variant = copy.deepcopy(variant)
            create_benchmark(test_variant, benchmark_dir)
            print(
                f"{variant.name}: priority={variant.priority} "
                f"pose={variant.object_pose[:3]} table_yaw={variant.scene_table_yaw_rad}"
            )
        return

    write_report(output_root, records, variants)
    for variant in variants:
        print(f"[variant] {variant.name}", flush=True)
        new_records = run_variant(
            variant=variant,
            output_root=output_root,
            checkpoint_path=checkpoint_path,
            smoke_horizon=args.smoke_horizon,
            task_horizon=args.task_horizon,
        )
        records.extend(new_records)
        done.update(record.variant["name"] for record in new_records)
        remaining = [candidate for candidate in variants if candidate.name not in done]
        write_report(output_root, records, remaining)
        full = next((record for record in new_records if record.stage == "full"), None)
        if full is not None:
            summary = full.summary or {}
            print(
                f"[done] {variant.name} move={summary.get('obj_delta_m')} "
                f"left_mid_min={summary.get('left_finger_midpoint_obj_dist_min_m')} "
                f"left_tcp_min={summary.get('left_tcp_obj_dist_min_m')} "
                f"contact={summary.get('left_finger_contact_any')}",
                flush=True,
            )
            if float(summary.get("obj_delta_m") or 0.0) > OBJECT_MOVE_THRESHOLD_M:
                print(f"[stop] object moved in {variant.name}", flush=True)
                break

    print(f"Report: {report_path(output_root, 'md')}", flush=True)


if __name__ == "__main__":
    main()
