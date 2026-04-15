#!/usr/bin/env python3
"""Run RBY1 Pick/PnP camera-view ablations.

Each variant keeps the same benchmark and frozen-base policy behavior while
changing the policy camera set/order through a dedicated eval config.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_rby1_eval import summarize_h5


DEFAULT_CHECKPOINT = REPO_ROOT / "ckpts" / "molmobot" / "MolmoBot-RBY1Multitask"
DEFAULT_BENCHMARK = REPO_ROOT / "benchmarks" / "rby1_pickpnp_benchmark"

CONFIG_PREFIX = "olmo.eval.configure_molmo_spaces:"


@dataclass
class ViewVariant:
    name: str
    eval_config_cls: str
    camera_names: list[str]
    reason: str


@dataclass
class ViewRunRecord:
    variant: dict[str, Any]
    stage: str
    output_dir: str
    command: list[str]
    returncode: int | None = None
    log_path: str | None = None
    h5_path: str | None = None
    videos: list[str] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None


def variants() -> list[ViewVariant]:
    return [
        ViewVariant(
            name="current_three_view_debug",
            eval_config_cls=CONFIG_PREFIX + "MolmoBotRBY1PickPnPViewDebugFrozenBaseEvalConfig",
            camera_names=["wrist_camera_r", "head_camera", "wrist_camera_l"],
            reason="Current frozen-base camera order with debug artifacts.",
        ),
        ViewVariant(
            name="left_focused_head_left",
            eval_config_cls=CONFIG_PREFIX + "MolmoBotRBY1PickPnPLeftFocusedFrozenBaseEvalConfig",
            camera_names=["head_camera", "wrist_camera_l"],
            reason="Head plus left wrist, matching left-arm pick relevance.",
        ),
        ViewVariant(
            name="left_wrist_first",
            eval_config_cls=CONFIG_PREFIX + "MolmoBotRBY1PickPnPLeftWristFirstFrozenBaseEvalConfig",
            camera_names=["wrist_camera_l", "head_camera"],
            reason="Left wrist first, then head.",
        ),
        ViewVariant(
            name="left_only",
            eval_config_cls=CONFIG_PREFIX + "MolmoBotRBY1PickPnPLeftOnlyFrozenBaseEvalConfig",
            camera_names=["wrist_camera_l"],
            reason="Left wrist only diagnostic.",
        ),
        ViewVariant(
            name="left_first_full",
            eval_config_cls=CONFIG_PREFIX + "MolmoBotRBY1PickPnPLeftFirstFullFrozenBaseEvalConfig",
            camera_names=["wrist_camera_l", "head_camera", "wrist_camera_r"],
            reason="Training views reordered with left wrist first.",
        ),
    ]


def eval_command(
    checkpoint_path: Path,
    benchmark_path: Path,
    eval_config_cls: str,
    task_horizon: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(REPO_ROOT / "launch_scripts" / "run_eval.py"),
        "--checkpoint_path",
        str(checkpoint_path),
        "--benchmark_path",
        str(benchmark_path),
        "--eval_config_cls",
        eval_config_cls,
        "--task_horizon",
        str(task_horizon),
        "--output_dir",
        str(output_dir),
        "--num_workers",
        "1",
    ]


def find_latest_h5(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.rglob("trajectories_batch_*.h5"))
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def find_videos(output_dir: Path) -> list[str]:
    return [str(path) for path in sorted(output_dir.rglob("*.mp4"))]


def materialize_single_episode_benchmark(
    source_benchmark: Path,
    output_root: Path,
    variant_name: str,
    episode_index: int,
) -> Path:
    """Create a benchmark directory containing one selected episode.

    The RBY1 pick benchmark currently has multiple geometry episodes. For view
    ablations we want one controlled episode per run so smoke/full results are
    fast and directly comparable.
    """
    source_benchmark = source_benchmark.resolve()
    source_json = source_benchmark / "benchmark.json"
    with source_json.open() as f:
        episodes = json.load(f)
    if not isinstance(episodes, list):
        raise ValueError(f"{source_json} must contain a list of episodes")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(
            f"episode_index={episode_index} is out of range for {source_json} "
            f"with {len(episodes)} episodes"
        )

    target = output_root / "_benchmarks" / f"{variant_name}_episode_{episode_index:02d}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark.json").write_text(json.dumps([episodes[episode_index]], indent=2) + "\n")

    for filename in ("custom_scene.xml", "custom_scene_metadata.json"):
        source_file = source_benchmark / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)
    return target


def run_record(record: ViewRunRecord) -> ViewRunRecord:
    output_dir = Path(record.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{record.stage}.log"
    record.log_path = str(log_path)
    record.started_at = time.time()

    env = os.environ.copy()
    env.setdefault("MLSPACES_RENDER_DEVICE_ID", "none")
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")

    with log_path.open("w") as log_file:
        proc = subprocess.run(
            record.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record.finished_at = time.time()
    record.returncode = proc.returncode

    h5_path = find_latest_h5(output_dir)
    if h5_path is not None:
        record.h5_path = str(h5_path)
        record.videos = find_videos(output_dir)
        summaries = summarize_h5(h5_path)
        record.summary = summaries[0] if summaries else None
    return record


def make_record(
    variant: ViewVariant,
    stage: str,
    output_root: Path,
    checkpoint_path: Path,
    benchmark_path: Path,
    horizon: int,
) -> ViewRunRecord:
    output_dir = output_root / variant.name / stage
    return ViewRunRecord(
        variant=asdict(variant),
        stage=stage,
        output_dir=str(output_dir),
        command=eval_command(
            checkpoint_path=checkpoint_path,
            benchmark_path=benchmark_path,
            eval_config_cls=variant.eval_config_cls,
            task_horizon=horizon,
            output_dir=output_dir,
        ),
    )


def report_path(output_root: Path, suffix: str) -> Path:
    return output_root / f"view_ablation_report.{suffix}"


def load_existing_records(output_root: Path) -> list[ViewRunRecord]:
    path = report_path(output_root, "json")
    if not path.exists():
        return []
    with path.open() as f:
        payload = json.load(f)

    records = []
    for item in payload.get("records", []):
        allowed = {field.name for field in ViewRunRecord.__dataclass_fields__.values()}
        records.append(ViewRunRecord(**{key: value for key, value in item.items() if key in allowed}))
    return records


def write_report(output_root: Path, records: list[ViewRunRecord]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path(output_root, "json").write_text(
        json.dumps(
            {
                "updated_at": time.time(),
                "records": [asdict(record) for record in records],
            },
            indent=2,
        )
        + "\n"
    )

    lines = ["# RBY1 View Ablation Report", ""]
    lines.append("## Completed")
    if not records:
        lines.append("- none")
    for record in records:
        summary = record.summary or {}
        lines.append(
            "- "
            f"{record.variant['name']} [{record.stage}] "
            f"cameras={record.variant['camera_names']} rc={record.returncode} "
            f"move={summary.get('obj_delta_m')} "
            f"min_left_tcp={summary.get('left_tcp_obj_dist_min_m')} "
            f"contact={summary.get('left_finger_contact_any')} "
            f"success={summary.get('success_any')} "
            f"h5={record.h5_path}"
        )
    report_path(output_root, "md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--benchmark_path", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output_root", default="eval_output/rby1_view_ablation")
    parser.add_argument("--smoke_horizon", type=int, default=25)
    parser.add_argument("--task_horizon", type=int, default=300)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--only", nargs="+", help="Variant names to run.")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    benchmark_path = Path(args.benchmark_path)
    output_root = Path(args.output_root)
    selected = variants()
    if args.only:
        wanted = set(args.only)
        selected = [variant for variant in selected if variant.name in wanted]
        missing = wanted - {variant.name for variant in selected}
        if missing:
            raise SystemExit(f"Unknown variants: {sorted(missing)}")

    if args.dry_run:
        for variant in selected:
            print(
                f"{variant.name}: {variant.camera_names} {variant.eval_config_cls} "
                f"episode_index={args.episode_index}"
            )
        return

    records: list[ViewRunRecord] = load_existing_records(output_root)
    for variant in selected:
        variant_benchmark_path = materialize_single_episode_benchmark(
            source_benchmark=benchmark_path,
            output_root=output_root,
            variant_name=variant.name,
            episode_index=args.episode_index,
        )
        if not args.skip_smoke:
            print(f"[smoke] {variant.name}", flush=True)
            smoke = make_record(
                variant,
                "smoke",
                output_root,
                checkpoint_path,
                variant_benchmark_path,
                args.smoke_horizon,
            )
            smoke = run_record(smoke)
            records.append(smoke)
            write_report(output_root, records)
            if smoke.returncode != 0 or smoke.h5_path is None:
                print(f"[smoke failed] {variant.name} rc={smoke.returncode}", flush=True)
                continue

        print(f"[full] {variant.name}", flush=True)
        full = make_record(
            variant,
            "full",
            output_root,
            checkpoint_path,
            variant_benchmark_path,
            args.task_horizon,
        )
        full = run_record(full)
        records.append(full)
        summary = full.summary or {}
        print(
            f"[done] {variant.name} rc={full.returncode} "
            f"move={summary.get('obj_delta_m')} "
            f"min_left_tcp={summary.get('left_tcp_obj_dist_min_m')}",
            flush=True,
        )
        write_report(output_root, records)

    print(f"Report: {report_path(output_root, 'md')}", flush=True)


if __name__ == "__main__":
    main()
