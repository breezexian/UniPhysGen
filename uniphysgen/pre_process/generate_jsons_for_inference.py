"""Generate four UniPhys-Bench inference manifests."""

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
        write_json,
    )
except ImportError:  # Support direct execution: python pre_process/...
    from _json_generation_common import (  # type: ignore
        add_part_label_to_name,
        has_valid_motion,
        load_entity_samples,
        write_json,
    )


OUTPUT_FILES = {
    "physics": "intrinsic_physics_part.json",
    "object_level": "intrinsic_physics_object.json",
    "motion": "kinematic_parameters.json",
    "group": "articulation_structure.json",
}


def generate_inference_manifests(
    data_root: str | Path,
    npz_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    entities, load_stats = load_entity_samples(data_root, npz_dir)
    manifests: dict[str, list[dict[str, Any]]] = {
        task_name: [] for task_name in OUTPUT_FILES
    }

    for entity in entities:
        manifests["object_level"].append(entity.object_sample)
        for sample in entity.part_samples:
            # Benchmark inference keeps every annotated part for intrinsic physics.
            manifests["physics"].append(sample)
            if not has_valid_motion(sample):
                continue
            manifests["motion"].append(sample)
            manifests["group"].append(
                add_part_label_to_name(sample, entity.part_label_to_name)
            )

    output_dir = Path(output_dir).expanduser().resolve()
    for task_name, filename in OUTPUT_FILES.items():
        write_json(output_dir / filename, manifests[task_name])

    summary = Counter(load_stats)
    summary.update({f"{task}_samples": len(items) for task, items in manifests.items()})
    print(json.dumps(dict(summary), indent=2, ensure_ascii=False))
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four task-specific batch manifests for UniPhysGen inference."
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
        help="Directory for the four generated inference JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_inference_manifests(args.data_root, args.npz_dir, args.output_dir)


if __name__ == "__main__":
    main()

