"""Generate four task-specific UniPhysGen training datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from ._json_generation_common import (
        add_part_label_to_name,
        has_valid_motion,
        load_entity_samples,
        mass_rate_is_valid,
        pass_check_is_valid,
        standard_dataset_info,
        write_json,
    )
except ImportError:  # Support direct execution: python pre_process/...
    from _json_generation_common import (  # type: ignore
        add_part_label_to_name,
        has_valid_motion,
        load_entity_samples,
        mass_rate_is_valid,
        pass_check_is_valid,
        standard_dataset_info,
        write_json,
    )


TASKS = {
    "physics": {
        "task_name": "physics",
        "dataset_name": "uniphys_40k_physics",
    },
    "kinematic_parameters": {
        "task_name": "motion",
        "dataset_name": "uniphys_40k_kinematic_parameters",
    },
    "articulation_structure": {
        "task_name": "group",
        "dataset_name": "uniphys_40k_articulation_structure",
    },
    "object_level": {
        "task_name": "object_level",
        "dataset_name": "uniphys_40k_object_level",
    },
}


def generate_training_datasets(
    data_root: str | Path,
    npz_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    entities, load_stats = load_entity_samples(data_root, npz_dir)
    datasets: dict[str, list[dict[str, Any]]] = {
        task_directory: [] for task_directory in TASKS
    }
    filter_stats: Counter[str] = Counter()

    for entity in entities:
        if mass_rate_is_valid(entity.object_sample):
            datasets["object_level"].append(entity.object_sample)
        else:
            filter_stats["object_level_dropped_mass_rate"] += 1

        for sample in entity.part_samples:
            if not mass_rate_is_valid(sample):
                filter_stats["physics_dropped_mass_rate"] += 1
            elif not pass_check_is_valid(sample):
                filter_stats["physics_dropped_pass_check"] += 1
            else:
                datasets["physics"].append(sample)

            # Motion-only tasks intentionally ignore mass_rate and pass_check.
            if has_valid_motion(sample):
                datasets["kinematic_parameters"].append(sample)
                datasets["articulation_structure"].append(
                    add_part_label_to_name(sample, entity.part_label_to_name)
                )
            else:
                filter_stats["motion_tasks_dropped_without_valid_motion"] += 1

    output_dir = Path(output_dir).expanduser().resolve()
    for task_directory, task_config in TASKS.items():
        task_output_dir = output_dir / task_directory
        write_json(task_output_dir / "train.json", datasets[task_directory])
        write_json(
            task_output_dir / "dataset_info.json",
            standard_dataset_info(
                task_config["dataset_name"], task_config["task_name"]
            ),
        )

    summary = Counter(load_stats)
    summary.update(filter_stats)
    summary.update(
        {f"{task}_samples": len(items) for task, items in datasets.items()}
    )
    print(json.dumps(dict(summary), indent=2, ensure_ascii=False))
    return datasets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate independent train.json and dataset_info.json files for "
            "the four UniPhysGen tasks."
        )
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Dataset root containing <entity>/annotations/object.json.",
    )
    parser.add_argument(
        "--npz_dir",
        required=True,
        help="Root passed as --output_dir to generate_npzs.py.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for four independent task directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_training_datasets(args.data_root, args.npz_dir, args.output_dir)


if __name__ == "__main__":
    main()

