import argparse
import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HF_REPO = "allenai/MolmoBot-DROID"
DEFAULT_RBY1_HF_REPO = "allenai/MolmoBot-RBY1Multitask"
DEFAULT_EVAL_CONFIG = "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"
DEFAULT_RBY1_SLICE_COUNT = 3


def _resolve_checkpoint_path(
    local_path: str | None,
    hf_repo: str | None,
    s3_path: str | None,
) -> tuple[str, str]:
    if local_path:
        return local_path, "local"

    if s3_path:
        from launch_scripts.serve_molmo import download_model_from_s3

        local_folder = s3_path.rstrip("/").split("/")[-1]
        download_dir = f"ckpts/molmobot/{local_folder}"
        return download_model_from_s3(s3_path, download_dir), "s3"

    if hf_repo:
        from launch_scripts.serve_molmo import download_model_from_hf

        local_folder = hf_repo.split("/")[-1]
        download_dir = f"ckpts/molmobot/{local_folder}"
        return download_model_from_hf(hf_repo, download_dir), "huggingface"

    raise ValueError("One of local_path, hf_repo, or s3_path must be provided")


def _get_git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _looks_like_rby1_eval(eval_config_cls: str) -> bool:
    return "MolmoBotRBY1" in eval_config_cls or "RBY1" in eval_config_cls


def _parse_episode_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not indices:
        return []
    return [int(chunk) for chunk in indices]


def _slice_descriptor(episode_indices: list[int] | None, slice_count: int | None) -> str:
    if episode_indices is not None:
        return "indices_" + "_".join(str(idx) for idx in episode_indices)
    if slice_count is not None:
        return f"first_{slice_count}"
    return "full"


def _normalize_benchmark_if_requested(args: argparse.Namespace) -> tuple[Path, Path | None]:
    benchmark_path = Path(args.benchmark_path).resolve()
    if not args.normalize_rby1_benchmark:
        return benchmark_path, None

    try:
        from launch_scripts.normalize_rby1_benchmark import normalize_benchmark
    except ModuleNotFoundError:
        from normalize_rby1_benchmark import normalize_benchmark

    episode_indices = _parse_episode_indices(args.episode_indices)
    slice_count = args.slice_count
    if episode_indices is None and slice_count is None:
        slice_count = DEFAULT_RBY1_SLICE_COUNT

    normalized_dir = (
        Path(args.normalized_benchmark_dir).resolve()
        if args.normalized_benchmark_dir
        else (
            Path("/tmp/molmobot_rby1_benchmark_slices")
            / benchmark_path.name
            / _slice_descriptor(episode_indices, slice_count)
        )
    )
    normalize_benchmark(
        benchmark_path,
        normalized_dir,
        episode_indices=episode_indices,
        slice_count=slice_count,
    )
    return normalized_dir, benchmark_path


def run_benchmark_smoke(args: argparse.Namespace) -> None:
    if args.hf_repo == DEFAULT_HF_REPO and _looks_like_rby1_eval(args.eval_config_cls):
        args.hf_repo = DEFAULT_RBY1_HF_REPO

    checkpoint_path, checkpoint_source = _resolve_checkpoint_path(
        local_path=args.local_path,
        hf_repo=args.hf_repo,
        s3_path=args.s3_path,
    )
    resolved_benchmark_path, source_benchmark_path = _normalize_benchmark_if_requested(args)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else (output_dir / "feasibility_manifest.json" if output_dir else None)
    )

    manifest = {
        "mode": "benchmark_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _get_git_commit(Path(__file__).resolve().parents[2]),
        "checkpoint_source": checkpoint_source,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "benchmark_path": str(resolved_benchmark_path),
        "source_benchmark_path": str(source_benchmark_path) if source_benchmark_path else None,
        "eval_config_cls": args.eval_config_cls,
        "task_horizon": args.task_horizon,
        "num_workers": args.num_workers,
        "use_filament": args.use_filament,
        "environment_light_intensity": args.environment_light_intensity,
        "normalize_rby1_benchmark": args.normalize_rby1_benchmark,
        "episode_indices": _parse_episode_indices(args.episode_indices),
        "slice_count": args.slice_count,
        "notes": args.notes,
    }
    if manifest_path is not None:
        _write_manifest(manifest_path, manifest)

    module_path, class_name = args.eval_config_cls.split(":", 1)
    eval_config_cls = getattr(importlib.import_module(module_path), class_name)

    from molmo_spaces.evaluation.eval_main import run_evaluation

    results = run_evaluation(
        eval_config_cls=eval_config_cls,
        benchmark_dir=resolved_benchmark_path,
        checkpoint_path=Path(checkpoint_path),
        task_horizon_steps=args.task_horizon,
        output_dir=str(output_dir) if output_dir else None,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_filament=args.use_filament,
        environment_light_intensity=args.environment_light_intensity,
    )

    print(f"Success rate: {results.success_rate:.1%}")
    print(f"Executed episodes: {results.total_count}")
    if manifest_path is not None:
        print(f"Manifest: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible MolmoBot feasibility checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "benchmark-smoke",
        help="Run a reproducible benchmark smoke test with a released MolmoBot checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = smoke.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--local-path", type=str, help="Local checkpoint directory")
    source_group.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO, help="Hugging Face repo to download")
    source_group.add_argument("--s3-path", type=str, help="S3 prefix containing a checkpoint")
    smoke.add_argument("--benchmark-path", type=str, required=True, help="Path to the MolmoSpaces JSON benchmark")
    smoke.add_argument("--eval-config-cls", type=str, default=DEFAULT_EVAL_CONFIG, help="Evaluation config class")
    smoke.add_argument(
        "--normalize-rby1-benchmark",
        action="store_true",
        help="Normalize released RBY1 benchmark metadata and optionally export a deterministic small slice",
    )
    smoke.add_argument(
        "--normalized-benchmark-dir",
        type=str,
        default=None,
        help="Optional output directory for the normalized/sliced benchmark",
    )
    smoke.add_argument(
        "--slice-count",
        type=int,
        default=None,
        help=f"Export the first N episodes when normalizing; defaults to {DEFAULT_RBY1_SLICE_COUNT} for RBY1 normalization",
    )
    smoke.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to export when normalizing",
    )
    smoke.add_argument("--task-horizon", type=int, default=600, help="Maximum number of steps per episode")
    smoke.add_argument("--output-dir", type=str, default=None, help="Output directory for eval results")
    smoke.add_argument("--manifest-path", type=str, default=None, help="Optional path for run metadata")
    smoke.add_argument("--num-workers", type=int, default=1, help="Number of eval workers")
    smoke.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    smoke.add_argument("--wandb-project", type=str, default="mjthor-online-eval", help="Weights & Biases project")
    smoke.add_argument("--use-filament", action="store_true", help="Use filament renderer")
    smoke.add_argument(
        "--environment-light-intensity",
        type=float,
        default=None,
        help="Default environmental light intensity for filament",
    )
    smoke.add_argument("--notes", type=str, default=None, help="Free-form notes to store in the manifest")
    smoke.set_defaults(func=run_benchmark_smoke)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
