"""Shared command-line runner for the four UniPhysGen inference tasks.

The public entry points are:

* ``inference_batch_intrinsic_physics_part.py`` (intrinsic physics, part level)
* ``inference_batch_intrinsic_physics_object.py`` (intrinsic physics, object level)
* ``inference_batch_kinematic_parameters.py`` (kinematic parameters)
* ``inference_batch_articulation_structure.py`` (articulation structure)

This module deliberately contains inference only and never computes metrics.
Each saved record embeds the untouched source sample so that evaluation can
derive ground truth without reopening dataset-specific annotation files.
Kinematic predictions are converted back to annotation/world units before they
are returned or saved, while the direct model output and reversible AABB
transform are retained alongside them.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    set_seed,
)

from uniphysgen.tuner.data.mm_plugin import (
    IMAGE_E_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_S_TOKEN,
    OBJECT_POINT_CLOUD_PLACEHOLDER,
    OBJECT_POINT_E_TOKEN,
    OBJECT_POINT_S_TOKEN,
    PART_POINT_CLOUD_PLACEHOLDER,
    PART_POINT_E_TOKEN,
    PART_POINT_S_TOKEN,
    UniPhysGenPlugin,
)
from uniphysgen.tuner.data.tasks import get_task_handler


TASKS = {"physics", "object_level", "motion", "group"}
TASK_TITLES = {
    "physics": "intrinsic physics (part level)",
    "object_level": "intrinsic physics (object level)",
    "motion": "kinematic parameters",
    "group": "articulation structure",
}


@dataclass(frozen=True)
class InferenceSample:
    """Resolved model inputs for one sample."""

    sample_id: str
    object_pcd: Path
    part_pcd: Optional[Path]
    image: Optional[Path]
    source_index: Optional[int] = None
    source_json: Optional[Path] = None
    source_sample: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CoordinateTransform:
    """Mapping from the model point-cloud frame back to the source frame.

    Inference uses the same deterministic transform as training, with stochastic
    augmentation disabled::

        point_model = (point_world - center) / scale - min_bound
    """

    center: np.ndarray
    scale: float
    min_bound: np.ndarray

    def point_to_model(self, value: Any) -> List[float]:
        point = _finite_array(value, length=3, field="point")
        model = (point - self.center) / self.scale - self.min_bound
        return [float(x) for x in model]

    def point_to_world(self, value: Any) -> List[float]:
        point = _finite_array(value, length=3, field="pivot")
        world = (point + self.min_bound) * self.scale + self.center
        return [float(x) for x in world]

    def distance_to_world(self, value: Any) -> List[float]:
        interval = _finite_array(value, length=2, field="range")
        return [float(x) for x in interval * self.scale]

    def distance_to_model(self, value: Any) -> List[float]:
        interval = _finite_array(value, length=2, field="range")
        return [float(x) for x in interval / self.scale]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exact source/model coordinate mapping used by inference."""

        min_bound = [float(x) for x in self.min_bound]
        return {
            "name": "source_to_aabb_0_2",
            "source_frame": "raw_annotation",
            "model_frame": "aabb_normalized_shifted",
            "center": [float(x) for x in self.center],
            "scale": float(self.scale),
            "min_bound": min_bound,
            "shift": [-x for x in min_bound],
            "point_source_to_model": "(point - center) / scale - min_bound",
            "point_model_to_source": "(point + min_bound) * scale + center",
            "distance_source_to_model": "distance / scale",
            "distance_model_to_source": "distance * scale",
        }


@dataclass(frozen=True)
class PreparedInputs:
    object_point_clouds: torch.Tensor
    part_point_clouds: Optional[torch.Tensor]
    images: Optional[torch.Tensor]
    transform: CoordinateTransform
    group_parts: List[Dict[str, Any]]


def _finite_array(value: Any, *, length: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain {length} finite numbers") from exc
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    return array


def spherical_axis_to_vector(axis: Mapping[str, Any]) -> List[float]:
    """Convert the model's spherical-axis representation to a unit vector."""

    theta = math.radians(float(axis["theta"]))
    phi = math.radians(float(axis["phi"]))
    vector = np.asarray(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )
    return [float(x) for x in vector]


def _unit_axis(value: Any) -> List[float]:
    if isinstance(value, Mapping):
        vector = np.asarray(spherical_axis_to_vector(value), dtype=np.float64)
    else:
        vector = _finite_array(value, length=3, field="axis")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("axis must be non-zero")
    return [float(x) for x in vector / norm]


def restore_motion_to_annotation_units(
    result: Mapping[str, Any], transform: CoordinateTransform
) -> Dict[str, Any]:
    """Return a motion prediction in the same units as raw annotations.

    The returned schema remains ``motion_type/axis/pivot/range``. No normalized
    duplicate, transform metadata, ground truth, or metric fields are retained.
    """

    motion_type = result.get("motion_type")
    if motion_type not in {"B", "C", None}:
        raise ValueError(f"motion_type must be 'B', 'C', or null; got {motion_type!r}")

    restored: Dict[str, Any] = {
        "motion_type": motion_type,
        "axis": None,
        "pivot": None,
        "range": None,
    }
    if result.get("axis") is not None:
        restored["axis"] = _unit_axis(result["axis"])
    if result.get("pivot") is not None:
        restored["pivot"] = transform.point_to_world(result["pivot"])
    if result.get("range") is not None:
        interval = _finite_array(result["range"], length=2, field="range")
        if motion_type == "C":
            interval = interval * (2.0 * math.pi)
            restored["range"] = [float(x) for x in interval]
        elif motion_type == "B":
            restored["range"] = transform.distance_to_world(interval)
        else:
            raise ValueError("range cannot be restored when motion_type is null")
    return restored


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the text between its first ``{`` and last ``}`` exactly once."""

    if "{" not in text or "}" not in text:
        return None
    value = text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(value)
    except Exception:
        return None


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _boolean_argument(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _resolve_media_path(
    value: Any, *, json_dir: Path, data_root: Optional[Path]
) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    base = data_root if data_root is not None else json_dir
    return (base / path).resolve(strict=False)


def _sample_id(item: Mapping[str, Any], index: int) -> str:
    explicit = _first_nonempty(item.get("sample_id"), item.get("id"))
    if explicit is not None:
        return str(explicit)

    item_index = item.get("index")
    if isinstance(item_index, Mapping):
        raw_pieces = [
            item_index.get("type_name"),
            item_index.get("entity"),
            item_index.get("label"),
        ]
        pieces = [
            str(piece) for piece in raw_pieces if piece is not None and piece != ""
        ]
        if pieces:
            return "-".join(pieces)
    elif item_index is not None:
        return str(item_index)
    return f"sample-{index:06d}"


def sample_from_json_item(
    item: Mapping[str, Any],
    *,
    index: int,
    task_name: str,
    json_dir: Path,
    data_root: Optional[Path],
    source_json: Optional[Path] = None,
) -> InferenceSample:
    """Read one item from the repository's fixed nested dataset schema."""

    object_level = item.get("object_level") or {}
    object_value = object_level.get("object_ply")
    if task_name == "object_level":
        part_value = None
        image_value = object_level.get("object_img")
    else:
        part_level = item.get("part_level") or {}
        part_value = part_level.get("part_ply")
        image_value = part_level.get("part_img")

    object_path = _resolve_media_path(
        object_value, json_dir=json_dir, data_root=data_root
    )
    part_path = _resolve_media_path(part_value, json_dir=json_dir, data_root=data_root)
    image_path = _resolve_media_path(
        image_value, json_dir=json_dir, data_root=data_root
    )
    if object_path is None:
        raise ValueError(f"batch item {index} is missing object point cloud path")
    if task_name != "object_level" and part_path is None:
        raise ValueError(f"batch item {index} is missing part point cloud path")

    return InferenceSample(
        sample_id=_sample_id(item, index),
        object_pcd=object_path,
        part_pcd=part_path,
        image=image_path,
        source_index=index,
        source_json=source_json,
        source_sample=deepcopy(dict(item)),
    )


def load_batch_samples(
    json_path: Path,
    *,
    task_name: str,
    data_root: Optional[Path],
) -> List[InferenceSample]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError("batch JSON must be a list")
    return [
        sample_from_json_item(
            item,
            index=index,
            task_name=task_name,
            json_dir=json_path.parent,
            data_root=data_root,
            source_json=json_path,
        )
        for index, item in enumerate(data)
    ]


def _validate_sample(sample: InferenceSample, *, use_image: bool) -> None:
    required = [("object point cloud", sample.object_pcd)]
    if sample.part_pcd is not None:
        required.append(("part point cloud", sample.part_pcd))
    if use_image and sample.image is not None:
        required.append(("image", sample.image))
    for kind, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"{kind} does not exist: {path}")


def _part_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def prepare_inputs(
    plugin: UniPhysGenPlugin,
    sample: InferenceSample,
    *,
    task_name: str,
    use_image: bool,
) -> PreparedInputs:
    """Load a sample with the deterministic training-time point transform."""

    _validate_sample(sample, use_image=use_image)
    load_group_metadata = task_name == "group"
    if load_group_metadata:
        object_points, object_colors, object_normals, object_npz = (
            plugin._load_points_colors_normals(str(sample.object_pcd), return_npz=True)
        )
    else:
        object_points, object_colors, object_normals = (
            plugin._load_points_colors_normals(str(sample.object_pcd))
        )
        object_npz = None

    object_points, center, scale = plugin._normalize_aabb(object_points)
    part_dict = None
    if task_name == "object_level":
        min_bound = object_points.min(axis=0).astype(np.float32)
    else:
        if sample.part_pcd is None:
            raise ValueError("part point cloud is required for this task")
        part_points, part_colors, part_normals = plugin._load_points_colors_normals(
            str(sample.part_pcd)
        )
        part_points = ((part_points - center) / (scale + 1e-12)).astype(np.float32)
        min_bound = np.minimum(
            object_points.min(axis=0), part_points.min(axis=0)
        ).astype(np.float32)
        part_points = (part_points - min_bound).astype(np.float32)
        part_dict = {
            "name": "part",
            "coord": part_points,
            "color": part_colors,
            "normal": np.asarray(part_normals, dtype=np.float32),
        }

    object_points = (object_points - min_bound).astype(np.float32)

    object_dict = {
        "name": "object",
        "coord": object_points,
        "color": object_colors,
        "normal": np.asarray(object_normals, dtype=np.float32),
    }
    shared_grid_min_coords = None
    if plugin.share_grid_origin:
        shared_grid_min_coords = [plugin._compute_shared_grid_min_coord(object_dict)]
    object_tensor = plugin._regularize_point_clouds(
        [object_dict], shared_grid_min_coords=shared_grid_min_coords
    )
    part_tensor: Optional[torch.Tensor]
    if task_name == "object_level":
        part_tensor = None
    else:
        assert part_dict is not None
        part_tensor = plugin._regularize_point_clouds(
            [part_dict], shared_grid_min_coords=shared_grid_min_coords
        )

    group_parts: List[Dict[str, Any]] = []
    if load_group_metadata and object_npz is not None:
        try:
            if "part_names" in object_npz and "part_centers" in object_npz:
                names = list(object_npz["part_names"])
                centers = np.asarray(object_npz["part_centers"], dtype=np.float32)
                centers = ((centers - center) / (scale + 1e-12) - min_bound).astype(
                    np.float32
                )
                count = min(len(names), int(centers.shape[0]))
                group_parts = [
                    {
                        "id": _part_name(names[i]),
                        "position": [float(f"{float(x):.4f}") for x in centers[i]],
                    }
                    for i in range(count)
                ]
        finally:
            close = getattr(object_npz, "close", None)
            if callable(close):
                close()

    images = None
    if use_image and sample.image is not None:
        images = plugin._regularize_images([[str(sample.image)]], use_image=True)

    return PreparedInputs(
        object_point_clouds=object_tensor,
        part_point_clouds=part_tensor,
        images=images,
        transform=CoordinateTransform(
            center=np.asarray(center, dtype=np.float64),
            scale=float(scale),
            min_bound=np.asarray(min_bound, dtype=np.float64),
        ),
        group_parts=group_parts,
    )


def build_model_prompt(
    *,
    task_name: str,
    use_part: bool,
    use_image: bool,
    spherical_axis: bool,
    group_parts: Sequence[Mapping[str, Any]],
    point_token: str,
    image_token: str,
) -> Tuple[str, str]:
    handler = get_task_handler(task_name)
    if task_name == "motion" and spherical_axis:
        prompt = handler.DEFAULT_PROMPT3
    else:
        prompt = handler.build_prompt()
    system = handler.build_system_prompt()

    if not use_part:
        prompt = prompt.replace(PART_POINT_CLOUD_PLACEHOLDER, "")
    if not use_image:
        prompt = prompt.replace(IMAGE_PLACEHOLDER, "")
    if task_name == "group":
        prompt = prompt.replace(
            "{{PART_LIST}}", json.dumps(list(group_parts), ensure_ascii=False)
        )

    prompt = prompt.replace(
        OBJECT_POINT_CLOUD_PLACEHOLDER,
        f"{OBJECT_POINT_S_TOKEN}{point_token}{OBJECT_POINT_E_TOKEN}",
    )
    if use_part:
        prompt = prompt.replace(
            PART_POINT_CLOUD_PLACEHOLDER,
            f"{PART_POINT_S_TOKEN}{point_token}{PART_POINT_E_TOKEN}",
        )
    if use_image:
        prompt = prompt.replace(
            IMAGE_PLACEHOLDER, f"{IMAGE_S_TOKEN}{image_token}{IMAGE_E_TOKEN}"
        )
    return prompt, system


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    system: str,
    prepared: PreparedInputs,
    args: argparse.Namespace,
) -> str:
    if args.seed >= 0:
        set_seed(args.seed)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    attention_mask = (input_ids != pad_id).long()

    object_point_clouds = prepared.object_point_clouds.to(
        model.device, dtype=torch.float32
    )
    part_point_clouds = prepared.part_point_clouds
    if part_point_clouds is not None:
        part_point_clouds = part_point_clouds.to(model.device, dtype=torch.float32)
    images = prepared.images
    if images is not None:
        images = images.to(model.device, dtype=torch.float32)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "object_point_clouds": object_point_clouds,
            "part_point_clouds": part_point_clouds,
            "images": images,
        },
        streamer=streamer,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    texts = []
    for text in streamer:
        print(text, end="", flush=True)
        texts.append(text)
    print()
    return "".join(texts)


def _load_model(
    args: argparse.Namespace, *, task_name: str
) -> Tuple[Any, Any, UniPhysGenPlugin]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer defines neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model.task_name = task_name
    if hasattr(model, "set_point_backbone_dtype"):
        model.set_point_backbone_dtype(torch.float32)
    if getattr(model, "point_backbone", None) is not None:
        for module in model.point_backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False

    point_config = getattr(model.config, "point_config", {}) or {}
    num_bins = int(point_config.get("num_bins", 400))
    plugin = UniPhysGenPlugin(
        point_token=args.point_token,
        image_token=args.image_token,
        num_bins=num_bins,
        do_augmentation=False,
        random_rotation=False,
        random_scaling=False,
        use_spherical_axis=args.spherical_axis,
        share_grid_origin=args.share_grid_origin,
        grid_sample_mode="test",
    )
    return model, tokenizer, plugin


def _postprocess_result(
    parsed: Mapping[str, Any],
    *,
    task_name: str,
    transform: CoordinateTransform,
) -> Dict[str, Any]:
    if task_name == "motion":
        return restore_motion_to_annotation_units(parsed, transform)
    return dict(parsed)


def infer_sample(
    sample: InferenceSample,
    *,
    task_name: str,
    model: Any,
    tokenizer: Any,
    plugin: UniPhysGenPlugin,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    use_image = bool(args.use_image and sample.image is not None)
    prepared = prepare_inputs(plugin, sample, task_name=task_name, use_image=use_image)
    prompt, system = build_model_prompt(
        task_name=task_name,
        use_part=prepared.part_point_clouds is not None,
        use_image=prepared.images is not None,
        spherical_axis=args.spherical_axis,
        group_parts=prepared.group_parts,
        point_token=args.point_token,
        image_token=args.image_token,
    )
    raw_response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        system=system,
        prepared=prepared,
        args=args,
    )
    parsed = parse_json_object(raw_response)
    if parsed is None:
        record = _base_record(sample, task_name=task_name)
        record.update(
            {
                "coordinate_frame": prepared.transform.to_dict(),
                "model_result": None,
                "result": None,
                "raw_response": raw_response,
                "error": "model response did not contain a valid JSON object",
            }
        )
        return record
    try:
        result = _postprocess_result(
            parsed, task_name=task_name, transform=prepared.transform
        )
    except Exception as exc:
        record = _base_record(sample, task_name=task_name)
        record.update(
            {
                "coordinate_frame": prepared.transform.to_dict(),
                "model_result": deepcopy(dict(parsed)),
                "result": None,
                "raw_response": raw_response,
                "error": f"invalid model response: {exc}",
            }
        )
        return record
    record = _base_record(sample, task_name=task_name)
    record.update(
        {
            "coordinate_frame": prepared.transform.to_dict(),
            "model_result": deepcopy(dict(parsed)),
            "result": result,
            "raw_response": raw_response,
        }
    )
    return record


def _base_record(sample: InferenceSample, *, task_name: str) -> Dict[str, Any]:
    """Build the stable, self-contained part of an inference record."""

    return {
        "schema_version": "uniphysgen.inference.v1",
        "sample_id": sample.sample_id,
        "task": task_name,
        "source_index": sample.source_index,
        "source_json": (
            str(sample.source_json) if sample.source_json is not None else None
        ),
        "source_sample": deepcopy(sample.source_sample),
        "inputs": {
            "object_pcd": str(sample.object_pcd),
            "part_pcd": str(sample.part_pcd) if sample.part_pcd is not None else None,
            "image": str(sample.image) if sample.image is not None else None,
        },
    }


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "sample"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def _single_sample_from_args(
    args: argparse.Namespace, *, task_name: str
) -> InferenceSample:
    if not args.object_pcd:
        raise ValueError("single-sample mode requires --object_pcd")
    if task_name != "object_level" and not args.part_pcd:
        raise ValueError("single-sample mode requires --part_pcd for this task")
    object_path = Path(args.object_pcd).expanduser().resolve(strict=False)
    part_value = None if task_name == "object_level" else args.part_pcd
    part_path = (
        Path(part_value).expanduser().resolve(strict=False) if part_value else None
    )
    image_path = (
        Path(args.image).expanduser().resolve(strict=False) if args.image else None
    )
    default_id = part_path.stem if part_path is not None else object_path.stem
    return InferenceSample(
        sample_id=args.sample_id or default_id,
        object_pcd=object_path,
        part_pcd=part_path,
        image=image_path,
    )


def _build_parser(*, task_name: str, program: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program,
        description=f"UniPhysGen inference: {TASK_TITLES[task_name]}",
    )
    parser.add_argument(
        "--model_path", required=True, help="Model directory or checkpoint"
    )
    parser.add_argument(
        "--output", help="Single result JSON, or consolidated batch result JSON"
    )
    parser.add_argument(
        "--output_dir",
        help="Optional directory for one self-contained JSON record per batch sample",
    )

    parser.add_argument(
        "--object_pcd", help="Object point cloud for single-sample mode"
    )
    parser.add_argument("--part_pcd", help="Part point cloud for single-sample mode")
    parser.add_argument("--image", help="Optional image for single-sample mode")
    parser.add_argument("--sample_id", help="Optional identifier for a single sample")

    parser.add_argument(
        "--input_json",
        help="Batch JSON; the top-level value must be a list",
    )
    parser.add_argument(
        "--data_root",
        help="Base directory for relative media paths in --input_json; defaults to the JSON directory",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Batch start index (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Batch end index (exclusive); -1 means all",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop the batch on the first failed sample",
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--point_token", default="<|point_pad|>")
    parser.add_argument("--image_token", default="<|image_pad|>")
    parser.add_argument(
        "--use_image",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        help="Use images when paths are present (also accepts true/false)",
    )
    parser.add_argument(
        "--spherical_axis",
        "--use_spherical_axis",
        dest="spherical_axis",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        help="Use the spherical-axis prompt; also accepts true/false (kinematic task only)",
    )
    parser.add_argument(
        "--share_grid_origin",
        dest="share_grid_origin",
        nargs="?",
        const=True,
        type=_boolean_argument,
        help="Use a shared object/part voxel-grid origin (default: true)",
    )
    parser.add_argument(
        "--no_share_grid_origin",
        dest="share_grid_origin",
        action="store_false",
        help="Disable the shared object/part voxel-grid origin",
    )
    parser.set_defaults(share_grid_origin=True)

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    return parser


def _slice_samples(
    samples: Sequence[InferenceSample], *, start: int, end: int
) -> Iterable[Tuple[int, InferenceSample]]:
    first = max(0, start)
    last = len(samples) if end < 0 else min(len(samples), end)
    if last < first:
        raise ValueError(f"invalid batch interval: start={start}, end={end}")
    return enumerate(samples[first:last], start=first)


def run_cli(task_name: str, *, program: str) -> None:
    """Run one of the four fixed inference entry points."""

    if task_name not in TASKS:
        raise ValueError(f"unknown inference task: {task_name}")
    parser = _build_parser(task_name=task_name, program=program)
    args = parser.parse_args()
    if args.spherical_axis and task_name != "motion":
        parser.error(
            "--spherical_axis is only valid for "
            "inference_batch_kinematic_parameters.py"
        )
    if args.input_json is None and args.output_dir:
        parser.error("--output_dir is only valid with --input_json")

    try:
        if args.input_json:
            input_json = Path(args.input_json).expanduser().resolve(strict=True)
            data_root = (
                Path(args.data_root).expanduser().resolve(strict=False)
                if args.data_root
                else None
            )
            samples = load_batch_samples(
                input_json, task_name=task_name, data_root=data_root
            )
        else:
            samples = [_single_sample_from_args(args, task_name=task_name)]
    except Exception as exc:
        parser.error(str(exc))

    selected = list(_slice_samples(samples, start=args.start, end=args.end))
    if not selected:
        parser.error("the selected batch interval contains no samples")

    model, tokenizer, plugin = _load_model(args, task_name=task_name)
    records: List[Dict[str, Any]] = []
    for progress, (source_index, sample) in enumerate(selected, start=1):
        print(
            f"[{progress}/{len(selected)}] {sample.sample_id} "
            f"(source index {source_index})",
            flush=True,
        )
        try:
            record = infer_sample(
                sample,
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                plugin=plugin,
                args=args,
            )
            if record.get("error"):
                print(f"  failed: {record['error']}", flush=True)
                if args.fail_fast:
                    raise ValueError(str(record["error"]))
        except Exception as exc:
            if args.fail_fast or not args.input_json:
                raise
            record = _base_record(sample, task_name=task_name)
            record.update(
                {
                    "coordinate_frame": None,
                    "model_result": None,
                    "result": None,
                    "raw_response": None,
                    "error": str(exc),
                }
            )
            print(f"  failed: {exc}", flush=True)
        records.append(record)

        if args.output_dir:
            filename = f"{source_index:06d}_{_safe_filename(sample.sample_id)}.json"
            _write_json(Path(args.output_dir) / filename, record)
        if args.output and args.input_json:
            _write_json(Path(args.output), records)

    if args.output and not args.input_json:
        _write_json(Path(args.output), records[0])
    elif not args.output and not args.output_dir:
        value: Any = records if args.input_json else records[0]
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
