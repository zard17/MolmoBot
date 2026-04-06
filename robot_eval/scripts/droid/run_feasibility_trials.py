import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "task"


def _load_tasks(task_args: list[str], tasks_file: str | None) -> list[str]:
    tasks = [task.strip() for task in task_args if task.strip()]
    if tasks_file:
        for line in Path(tasks_file).read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tasks.append(stripped)
    deduped: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    if not deduped:
        raise ValueError("No tasks provided. Use --task and/or --tasks-file.")
    return deduped


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small Franka/DROID MolmoBot feasibility suite and aggregate the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-host", required=True, help="Franka or DROID NUC IP/hostname")
    parser.add_argument("--wrist-camera-id", required=True, type=int, help="Wrist ZED serial")
    parser.add_argument("--exo-camera-id", required=True, type=int, help="Exo ZED serial")
    parser.add_argument("--policy-host", default="0.0.0.0", help="Policy websocket host")
    parser.add_argument("--policy-port", type=int, default=8000, help="Policy websocket port")
    parser.add_argument("--output-dir", default="robot_eval/outputs/feasibility", help="Suite output directory")
    parser.add_argument("--tasks-file", default=None, help="Text file with one task per line")
    parser.add_argument("--task", action="append", default=[], help="Task to run. Repeat for multiple tasks.")
    parser.add_argument("--policy-dt", type=float, default=0.066, help="Control period in seconds")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Action interpolation scale")
    parser.add_argument("--image-width", type=int, default=640, help="Camera input width")
    parser.add_argument("--image-height", type=int, default=360, help="Camera input height")
    parser.add_argument(
        "--image-method",
        choices=["downsample", "center_pad", "center_crop", "none"],
        default="downsample",
        help="Image resize mode passed to the DROID client",
    )
    parser.add_argument("--disable-autohome", action="store_true", help="Disable homing between trials")
    parser.add_argument("--notes", default=None, help="Free-form notes stored in the suite manifest")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = _load_tasks(args.task, args.tasks_file)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "franka_droid_real_trials",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot_host": args.robot_host,
        "wrist_camera_id": args.wrist_camera_id,
        "exo_camera_id": args.exo_camera_id,
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "policy_dt": args.policy_dt,
        "action_scale": args.action_scale,
        "image_input": {
            "width": args.image_width,
            "height": args.image_height,
            "method": args.image_method,
        },
        "autohome_enabled": not args.disable_autohome,
        "tasks": tasks,
        "notes": args.notes,
    }
    _write_json(output_dir / "suite_manifest.json", manifest)

    results: list[dict[str, Any]] = []
    base_cmd = [
        sys.executable,
        "scripts/droid/run_policy.py",
        f"robot.robot_host={args.robot_host}",
        f"robot.cameras.wrist_camera.id={args.wrist_camera_id}",
        f"robot.cameras.exo_camera_1.id={args.exo_camera_id}",
        f"policy_host={args.policy_host}",
        f"policy_port={args.policy_port}",
        f"policy_dt={args.policy_dt}",
        f"action_scale={args.action_scale}",
        f"image_input.width={args.image_width}",
        f"image_input.height={args.image_height}",
        f"image_input.method={args.image_method}",
        f"autohome.enabled={'false' if args.disable_autohome else 'true'}",
    ]

    repo_root = Path(__file__).resolve().parents[3]
    robot_eval_dir = repo_root / "robot_eval"

    for index, task in enumerate(tasks, start=1):
        trial_dir = output_dir / f"trial_{index:02d}_{_slugify(task)}"
        cmd = [
            *base_cmd,
            f"task={task}",
            f"hydra.run.dir={trial_dir}",
        ]
        rendered_cmd = subprocess.list2cmdline(cmd)
        print(f"[trial {index}/{len(tasks)}] {task}")
        print(rendered_cmd)

        if args.dry_run:
            results.append(
                {
                    "task": task,
                    "trial_dir": str(trial_dir),
                    "status": "dry_run",
                    "command": rendered_cmd,
                }
            )
            continue

        completed = subprocess.run(cmd, cwd=robot_eval_dir, check=False)
        info = _read_json(trial_dir / "info.json")
        results.append(
            {
                "task": task,
                "trial_dir": str(trial_dir),
                "returncode": completed.returncode,
                "info": info,
            }
        )

    successes = [
        result["info"]["success"]
        for result in results
        if result.get("info") is not None and "success" in result["info"]
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_trials": len(results),
        "n_completed": sum(1 for result in results if result.get("info") is not None),
        "n_successes": sum(bool(success) for success in successes),
        "success_rate": (sum(bool(success) for success in successes) / len(successes)) if successes else None,
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
