import argparse
import json
from pathlib import Path

TASK_CLASS_REWRITES = {
    "mujoco_thor.tasks.opening_tasks.DoorOpeningTask": (
        "molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
        "door_opening",
    ),
    "mujoco_thor.tasks.opening_tasks.OpeningTask": (
        "molmo_spaces.tasks.opening_tasks.OpeningTask",
        "open",
    ),
}


def normalize_episode(episode: dict) -> dict:
    episode = json.loads(json.dumps(episode))
    task = episode.setdefault("task", {})
    task_cls = task.get("task_cls")
    if task_cls in TASK_CLASS_REWRITES:
        normalized_cls, task_type = TASK_CLASS_REWRITES[task_cls]
        task["task_cls"] = normalized_cls
        task.setdefault("task_type", task_type)
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize released RBY1 benchmark JSON for current molmo_spaces eval")
    parser.add_argument("--input-dir", required=True, help="Directory containing benchmark.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write normalized benchmark.json")
    parser.add_argument("--episode-idx", type=int, default=None, help="Optional single episode index to export")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    episodes = json.load(open(input_dir / "benchmark.json"))

    if args.episode_idx is not None:
        episodes = [episodes[args.episode_idx]]

    normalized = [normalize_episode(ep) for ep in episodes]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark.json", "w") as f:
        json.dump(normalized, f)

    metadata_path = input_dir / "benchmark_metadata.json"
    if metadata_path.exists():
        metadata = json.load(open(metadata_path))
        metadata["num_episodes"] = len(normalized)
        with open(output_dir / "benchmark_metadata.json", "w") as f:
            json.dump(metadata, f)

    print(output_dir)


if __name__ == "__main__":
    main()
