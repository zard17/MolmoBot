import argparse
import json
from pathlib import Path
from typing import Iterable

TASK_CLASS_REWRITES = {
    "mujoco_thor.tasks.opening_tasks.DoorOpeningTask": (
        "molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
        "door_opening",
    ),
    "mujoco_thor.tasks.opening_tasks.OpeningTask": (
        "molmo_spaces.tasks.opening_tasks.OpeningTask",
        "open",
    ),
    "mujoco_thor.tasks.pick_task.PickTask": (
        "molmo_spaces.tasks.pick_task.PickTask",
        "pick",
    ),
    "mujoco_thor.tasks.pick_and_place_task.PickAndPlaceTask": (
        "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
        "pick_and_place",
    ),
}


def normalize_episode(episode: dict) -> dict:
    episode = json.loads(json.dumps(episode))
    task = episode.setdefault("task", {})
    task_cls = task.get("task_cls")
    if task_cls in TASK_CLASS_REWRITES:
        normalized_cls, task_type = TASK_CLASS_REWRITES[task_cls]
        task["task_cls"] = normalized_cls
        task["task_type"] = task_type
    return episode


def select_episodes(
    episodes: list[dict], *, episode_indices: Iterable[int] | None = None, slice_count: int | None = None
) -> list[dict]:
    if episode_indices is not None:
        selected = []
        for idx in episode_indices:
            selected.append(episodes[idx])
        return selected

    if slice_count is not None:
        return episodes[:slice_count]

    return episodes


def normalize_benchmark(
    input_dir: Path,
    output_dir: Path,
    *,
    episode_indices: list[int] | None = None,
    slice_count: int | None = None,
) -> Path:
    with open(input_dir / "benchmark.json") as f:
        episodes = json.load(f)

    selected = select_episodes(episodes, episode_indices=episode_indices, slice_count=slice_count)
    normalized = [normalize_episode(ep) for ep in selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark.json", "w") as f:
        json.dump(normalized, f)

    metadata_path = input_dir / "benchmark_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["num_episodes"] = len(normalized)
        with open(output_dir / "benchmark_metadata.json", "w") as f:
            json.dump(metadata, f)

    return output_dir


def _parse_episode_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not indices:
        return []
    return [int(chunk) for chunk in indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize released RBY1 benchmark JSON for current molmo_spaces eval"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing benchmark.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write normalized benchmark.json")
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Backward-compatible alias for exporting a single episode index",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to export in order",
    )
    parser.add_argument(
        "--slice-count",
        type=int,
        default=None,
        help="Export the first N episodes after normalization",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    episode_indices = _parse_episode_indices(args.episode_indices)
    if args.episode_idx is not None:
        episode_indices = [args.episode_idx]

    normalize_benchmark(
        input_dir,
        output_dir,
        episode_indices=episode_indices,
        slice_count=args.slice_count,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
