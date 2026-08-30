"""Shared data loading and filtering for UniPhysGen JSON generation."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PART_ANNOTATION_PATTERN = re.compile(r"^part_(\d+)\.json$", flags=re.IGNORECASE)
MOTION_TYPES = ("B", "C")
MASS_RATE_LIMIT = 2.0


@dataclass(frozen=True)
class EntitySamples:
    """Prepared samples belonging to one object entity."""

    entity_id: str
    object_sample: dict[str, Any]
    part_samples: tuple[dict[str, Any], ...]
    part_label_to_name: dict[str, str]


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _part_id(path: Path) -> str:
    match = PART_ANNOTATION_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"expected part_<id>.json, got {path.name!r}")
    return match.group(1)


def _part_sort_key(path: Path) -> tuple[int, str]:
    part_id = _part_id(path)
    return int(part_id), part_id


def _relative_media_path(path: Path, npz_root: Path) -> str:
    return path.relative_to(npz_root).as_posix()


def _find_npz_entity_dir(npz_root: Path, relative_entity_dir: Path) -> Path | None:
    """Resolve both the current nested output and a flat compatible layout."""
    entity_root = npz_root / relative_entity_dir
    candidates = (entity_root / "npzs", entity_root)
    for candidate in candidates:
        if (candidate / "model.npz").is_file():
            return candidate
    return None


def _object_level_from_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    """Accept the release format and the earlier wrapped object-level format."""
    nested = annotation.get("object_level")
    if isinstance(nested, dict):
        return deepcopy(nested)
    return deepcopy(annotation)


def _part_label_to_name(
    annotations: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for filename_id, annotation in annotations:
        index = annotation.get("index")
        label = index.get("label") if isinstance(index, Mapping) else filename_id
        if label is None or str(label).strip() == "":
            label = filename_id

        part_level = annotation.get("part_level")
        part_name = (
            part_level.get("part_name") if isinstance(part_level, Mapping) else None
        )
        if part_name is None or str(part_name).strip() == "":
            continue
        mapping.setdefault(str(label), str(part_name))
    return mapping


def _make_object_sample(
    *,
    entity_id: str,
    object_level: dict[str, Any],
    part_annotations: list[tuple[str, dict[str, Any]]],
    part_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    # The earlier object-level generator selected one part record per entity.
    # Retain that superset schema for compatibility with the training converter.
    if part_samples:
        sample = deepcopy(part_samples[0])
    elif part_annotations:
        sample = deepcopy(part_annotations[0][1])
    else:
        sample = {
            "index": {
                "type_name": "default",
                "entity": entity_id,
                "label": "object",
            },
            "part_level": {},
            "basic_info": {},
            "kinematic_info": {},
        }
    sample["object_level"] = deepcopy(object_level)
    return sample


def load_entity_samples(
    data_root: str | Path,
    npz_root: str | Path,
) -> tuple[list[EntitySamples], Counter[str]]:
    """Load and merge released annotations with generated NPZ paths."""
    data_root = Path(data_root).expanduser().resolve()
    npz_root = Path(npz_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {data_root}")
    if not npz_root.is_dir():
        raise FileNotFoundError(f"NPZ root is not a directory: {npz_root}")

    object_annotations = sorted(data_root.rglob("annotations/object.json"))
    if not object_annotations:
        raise FileNotFoundError(
            f"no annotations/object.json files were found below: {data_root}"
        )

    entities: list[EntitySamples] = []
    stats: Counter[str] = Counter(entities_discovered=len(object_annotations))

    for object_annotation_path in object_annotations:
        annotation_dir = object_annotation_path.parent
        entity_dir = annotation_dir.parent
        relative_entity_dir = entity_dir.relative_to(data_root)
        entity_id = entity_dir.name

        try:
            object_annotation = _read_json_object(object_annotation_path)
            object_level = _object_level_from_annotation(object_annotation)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            stats["invalid_object_annotations"] += 1
            _warn(f"skipping {entity_id}: cannot read {object_annotation_path}: {exc}")
            continue

        entity_npz_dir = _find_npz_entity_dir(npz_root, relative_entity_dir)
        if entity_npz_dir is None:
            stats["missing_object_npzs"] += 1
            _warn(f"skipping {entity_id}: model.npz was not found below the NPZ root")
            continue

        object_npz_path = entity_npz_dir / "model.npz"
        object_level["object_ply"] = _relative_media_path(object_npz_path, npz_root)

        part_annotations: list[tuple[str, dict[str, Any]]] = []
        part_annotation_paths = sorted(
            (
                path
                for path in annotation_dir.iterdir()
                if path.is_file() and PART_ANNOTATION_PATTERN.fullmatch(path.name)
            ),
            key=_part_sort_key,
        )
        for part_annotation_path in part_annotation_paths:
            part_id = _part_id(part_annotation_path)
            try:
                annotation = _read_json_object(part_annotation_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                stats["invalid_part_annotations"] += 1
                _warn(f"skipping {entity_id}/part_{part_id}: {exc}")
                continue
            if not isinstance(annotation.get("part_level"), dict):
                stats["invalid_part_annotations"] += 1
                _warn(f"skipping {entity_id}/part_{part_id}: part_level must be an object")
                continue
            part_annotations.append((part_id, annotation))

        part_samples: list[dict[str, Any]] = []
        for part_id, annotation in part_annotations:
            part_npz_path = entity_npz_dir / f"{part_id}.npz"
            if not part_npz_path.is_file():
                stats["missing_part_npzs"] += 1
                _warn(f"skipping {entity_id}/part_{part_id}: {part_npz_path} is missing")
                continue

            sample = deepcopy(annotation)
            sample["object_level"] = deepcopy(object_level)
            sample["part_level"]["part_ply"] = _relative_media_path(
                part_npz_path, npz_root
            )
            part_samples.append(sample)

        entities.append(
            EntitySamples(
                entity_id=entity_id,
                object_sample=_make_object_sample(
                    entity_id=entity_id,
                    object_level=object_level,
                    part_annotations=part_annotations,
                    part_samples=part_samples,
                ),
                part_samples=tuple(part_samples),
                part_label_to_name=_part_label_to_name(part_annotations),
            )
        )
        stats["entities_loaded"] += 1
        stats["parts_loaded"] += len(part_samples)

    return entities, stats


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, set, str, bytes)):
        return len(value) > 0
    return True


def has_valid_motion(sample: Mapping[str, Any]) -> bool:
    """Match the original motion/group selection rule for B/C annotations."""
    kinematic_info = sample.get("kinematic_info")
    if not isinstance(kinematic_info, Mapping):
        return False
    motion_info = kinematic_info.get("motion_info")
    if not isinstance(motion_info, Mapping):
        return False
    return any(_is_non_empty(motion_info.get(key)) for key in MOTION_TYPES)


def mass_rate_is_valid(sample: Mapping[str, Any]) -> bool:
    object_level = sample.get("object_level")
    if not isinstance(object_level, Mapping):
        return False
    value = object_level.get("mass_rate")
    if isinstance(value, bool):
        return False
    try:
        return float(value) <= MASS_RATE_LIMIT
    except (TypeError, ValueError):
        return False


def pass_check_is_valid(sample: Mapping[str, Any]) -> bool:
    basic_info = sample.get("basic_info")
    return isinstance(basic_info, Mapping) and basic_info.get("pass_check") is True


def add_part_label_to_name(
    sample: Mapping[str, Any], mapping: Mapping[str, str]
) -> dict[str, Any]:
    output = deepcopy(dict(sample))
    part_level = output.get("part_level")
    if not isinstance(part_level, dict):
        raise TypeError("part_level must be an object")
    part_level["part_label_to_name"] = dict(mapping)
    return output


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON atomically so an interrupted run does not leave a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def standard_dataset_info(dataset_name: str, task_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": "train.json",
            "formatting": "physmeshllm",
            "task_name": task_name,
            "use_image": False,
            "columns": {
                "object_info": "object_level",
                "part_info": "part_level",
                "basic_info": "basic_info",
                "kinematic_info": "kinematic_info",
            },
        }
    }
