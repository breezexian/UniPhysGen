"""Typed configuration and path resolution for the UniPhys pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """Raised when a pipeline configuration is missing or invalid."""


def _expand_path(value: str | Path | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve(strict=False)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section '{name}' must be a mapping.")
    return value


def _string_tuple(value: Any, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ConfigError("Expected a comma-separated string or a list of stage names.")


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ConfigError(f"Configuration value '{name}' must be a boolean.")


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and derived input/output locations."""

    name: str = "ABO"
    type_name: str = "4"
    input_root: Path = PROJECT_ROOT / "samples"
    output_root: Path = PROJECT_ROOT / "outputs"
    mesh_root_override: Path | None = None
    render_root_override: Path | None = None
    decomposition_root_override: Path | None = None
    gpt_output_root_override: Path | None = None
    extensions: tuple[str, ...] = (".glb", ".obj")

    @property
    def mesh_root(self) -> Path:
        return self.mesh_root_override or self.input_root / self.name / self.type_name

    @property
    def dataset_output_root(self) -> Path:
        return self.output_root / self.name

    @property
    def render_root(self) -> Path:
        return self.render_root_override or self.dataset_output_root / "render_res"

    @property
    def decomposition_root(self) -> Path:
        return (
            self.decomposition_root_override
            or self.dataset_output_root / "decomposition_res" / self.type_name
        )

    @property
    def gpt_output_root(self) -> Path:
        return (
            self.gpt_output_root_override
            or self.dataset_output_root / "gpt_output" / self.type_name
        )


DEFAULT_STAGES = (
    "render",
    "decompose",
    "export_parts",
    "annotate_basic",
    "build_kinematic_graph",
    "propose_kinematics",
    "render_axes",
    "annotate_revolute",
    "annotate_prismatic",
    "generate_mujoco",
    "validate_basic",
    "validate_simulation",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Algorithm options and execution policy."""

    stages: tuple[str, ...] = DEFAULT_STAGES
    render_num: int = 8
    show_part: bool = False
    max_merge_num: int = 10
    cmp_iou_type: str = "area_type"
    refine_sam_masks: bool = True
    run_sam2_if_missing: bool = True
    max_faces: int = 500_000
    faces_per_batch: int = 30_000
    max_parts: int = 50
    cluster_search_min: int = 3
    cluster_search_max: int = 15
    part_refine_timeout_seconds: int = 60
    merge_timeout_seconds: int = 30
    part_refine_after_merge_use_relative_threshold: bool = False
    post_process_use_relative_threshold: bool = False
    cleanup_intermediates: bool = True
    remove_part_glbs_after_render: bool = True
    resume: bool = True
    adopt_existing_outputs: bool = True
    fail_fast: bool = False
    workers: int = 1
    start: int = 0
    end: int | None = None

    def validate(self) -> None:
        if not self.stages:
            raise ConfigError("pipeline.stages cannot be empty.")
        if self.render_num < 1:
            raise ConfigError("pipeline.render_num must be at least 1.")
        if self.max_merge_num < 1:
            raise ConfigError("pipeline.max_merge_num must be at least 1.")
        if self.cmp_iou_type not in {"area_type", "face_type"}:
            raise ConfigError(
                "pipeline.cmp_iou_type must be 'area_type' or 'face_type'."
            )
        if self.workers < 1:
            raise ConfigError("pipeline.workers must be at least 1.")
        if (
            self.cluster_search_min < 1
            or self.cluster_search_max < self.cluster_search_min
        ):
            raise ConfigError(
                "pipeline cluster search bounds must satisfy 1 <= min <= max."
            )
        if self.part_refine_timeout_seconds < 1 or self.merge_timeout_seconds < 1:
            raise ConfigError("Pipeline timeout values must be positive integers.")
        if self.start < 0:
            raise ConfigError("pipeline.start cannot be negative.")
        if self.end is not None and self.end < self.start:
            raise ConfigError("pipeline.end cannot be smaller than pipeline.start.")


@dataclass(frozen=True)
class RuntimeConfig:
    """External executables and model locations."""

    blender: str = "blender"
    env_dir: Path | None = None
    sam_python: Path | None = None
    partfield_python: Path | None = None
    sam_checkpoint: Path = PROJECT_ROOT / "utils/sam2/checkpoints/sam2_hiera_large.pt"
    partfield_checkpoint: Path = (
        PROJECT_ROOT / "utils/partfield/model/model_objaverse.ckpt"
    )
    partfield_config: Path = PROJECT_ROOT / "utils/partfield/configs/final/demo.yaml"
    command_timeout_seconds: int | None = None
    gpu: str | None = "0"
    lock_stale_seconds: int = 86_400

    @property
    def resolved_sam_python(self) -> Path:
        if self.sam_python is not None:
            return self.sam_python
        if self.env_dir is not None:
            return self.env_dir / "sam/bin/python"
        return Path(sys.executable)

    @property
    def resolved_partfield_python(self) -> Path:
        if self.partfield_python is not None:
            return self.partfield_python
        if self.env_dir is not None:
            return self.env_dir / "partfield/bin/python"
        return Path(sys.executable)


@dataclass(frozen=True)
class GPTConfig:
    """Configuration passed to the existing GPT annotation functions."""

    model: str = "gpt-5"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    basic_prompt: Path = PROJECT_ROOT / "utils/gpt/sys_basic_prompt.txt"
    axis_prompt: Path = PROJECT_ROOT / "utils/gpt/sys_axis_prompt.txt"

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ConfigError(
                f"Environment variable '{self.api_key_env}' is required for GPT stages."
            )
        return value

    def as_legacy_dict(self) -> dict[str, str | None]:
        return {
            "use_model": self.model,
            "base_url": self.base_url or os.environ.get("OPENAI_BASE_URL"),
            "api_key": self.api_key(),
            "basic_sys_prompt_pth": str(self.basic_prompt),
            "axis_sys_prompt_pth": str(self.axis_prompt),
        }


@dataclass(frozen=True)
class AppConfig:
    """Complete immutable application configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    source: Path | None = None

    def validate(self) -> None:
        self.pipeline.validate()
        if not self.dataset.extensions:
            raise ConfigError("dataset.extensions cannot be empty.")
        if (
            self.runtime.command_timeout_seconds is not None
            and self.runtime.command_timeout_seconds < 1
        ):
            raise ConfigError(
                "runtime.command_timeout_seconds must be positive or null."
            )
        if self.runtime.lock_stale_seconds < 1:
            raise ConfigError("runtime.lock_stale_seconds must be a positive integer.")

    def execution_fingerprint(self) -> str:
        """Return a stable digest without serializing secret values."""

        payload = {
            "dataset": {
                "name": self.dataset.name,
                "type_name": self.dataset.type_name,
                "mesh_root": str(self.dataset.mesh_root),
                "render_root": str(self.dataset.render_root),
                "decomposition_root": str(self.dataset.decomposition_root),
                "gpt_output_root": str(self.dataset.gpt_output_root),
            },
            "pipeline": {
                key: value
                for key, value in asdict(self.pipeline).items()
                if key not in {"stages", "workers", "start", "end", "fail_fast"}
            },
            "runtime": {
                "blender": self.runtime.blender,
                "sam_python": str(self.runtime.resolved_sam_python),
                "partfield_python": str(self.runtime.resolved_partfield_python),
                "sam_checkpoint": str(self.runtime.sam_checkpoint),
                "partfield_checkpoint": str(self.runtime.partfield_checkpoint),
            },
            "gpt": {
                "model": self.gpt.model,
                "base_url": self.gpt.base_url,
                "api_key_env": self.gpt.api_key_env,
                "basic_prompt": str(self.gpt.basic_prompt),
                "axis_prompt": str(self.gpt.axis_prompt),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def with_overrides(
        self,
        *,
        stages: Sequence[str] | None = None,
        workers: int | None = None,
        start: int | None = None,
        end: int | None = None,
        resume: bool | None = None,
        fail_fast: bool | None = None,
        gpu: str | None = None,
    ) -> "AppConfig":
        pipeline = replace(
            self.pipeline,
            stages=tuple(stages) if stages is not None else self.pipeline.stages,
            workers=workers if workers is not None else self.pipeline.workers,
            start=start if start is not None else self.pipeline.start,
            end=end if end is not None else self.pipeline.end,
            resume=resume if resume is not None else self.pipeline.resume,
            fail_fast=fail_fast if fail_fast is not None else self.pipeline.fail_fast,
        )
        runtime = replace(
            self.runtime, gpu=gpu if gpu is not None else self.runtime.gpu
        )
        updated = replace(self, pipeline=pipeline, runtime=runtime)
        updated.validate()
        return updated


def _load_raw(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on user environment
                raise ConfigError(
                    "PyYAML is required for YAML configuration files. "
                    "Install the project dependencies or use JSON."
                ) from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("The configuration root must be a mapping.")
    return data


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML/JSON configuration, resolving paths from the repository root."""

    config_path = _expand_path(path) if path is not None else None
    raw: Mapping[str, Any] = _load_raw(config_path) if config_path else {}

    dataset_raw = _mapping(raw.get("dataset"), "dataset")
    pipeline_raw = _mapping(raw.get("pipeline"), "pipeline")
    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    gpt_raw = _mapping(raw.get("gpt"), "gpt")

    dataset = DatasetConfig(
        name=str(dataset_raw.get("name", "ABO")),
        type_name=str(dataset_raw.get("type_name", "4")),
        input_root=_expand_path(dataset_raw.get("input_root", "samples"))
        or PROJECT_ROOT / "samples",
        output_root=_expand_path(dataset_raw.get("output_root", "outputs"))
        or PROJECT_ROOT / "outputs",
        mesh_root_override=_expand_path(dataset_raw.get("mesh_root")),
        render_root_override=_expand_path(dataset_raw.get("render_root")),
        decomposition_root_override=_expand_path(dataset_raw.get("decomposition_root")),
        gpt_output_root_override=_expand_path(dataset_raw.get("gpt_output_root")),
        extensions=tuple(
            (ext if str(ext).startswith(".") else f".{ext}").lower()
            for ext in _string_tuple(
                dataset_raw.get("extensions"), default=(".glb", ".obj")
            )
        ),
    )

    defaults = PipelineConfig()
    pipeline = PipelineConfig(
        stages=_string_tuple(pipeline_raw.get("stages"), default=DEFAULT_STAGES),
        render_num=int(pipeline_raw.get("render_num", defaults.render_num)),
        show_part=_boolean(
            pipeline_raw.get("show_part", defaults.show_part), name="pipeline.show_part"
        ),
        max_merge_num=int(pipeline_raw.get("max_merge_num", defaults.max_merge_num)),
        cmp_iou_type=str(pipeline_raw.get("cmp_iou_type", defaults.cmp_iou_type)),
        refine_sam_masks=_boolean(
            pipeline_raw.get("refine_sam_masks", defaults.refine_sam_masks),
            name="pipeline.refine_sam_masks",
        ),
        run_sam2_if_missing=_boolean(
            pipeline_raw.get("run_sam2_if_missing", defaults.run_sam2_if_missing),
            name="pipeline.run_sam2_if_missing",
        ),
        max_faces=int(pipeline_raw.get("max_faces", defaults.max_faces)),
        faces_per_batch=int(
            pipeline_raw.get("faces_per_batch", defaults.faces_per_batch)
        ),
        max_parts=int(pipeline_raw.get("max_parts", defaults.max_parts)),
        cluster_search_min=int(
            pipeline_raw.get("cluster_search_min", defaults.cluster_search_min)
        ),
        cluster_search_max=int(
            pipeline_raw.get("cluster_search_max", defaults.cluster_search_max)
        ),
        part_refine_timeout_seconds=int(
            pipeline_raw.get(
                "part_refine_timeout_seconds", defaults.part_refine_timeout_seconds
            )
        ),
        merge_timeout_seconds=int(
            pipeline_raw.get("merge_timeout_seconds", defaults.merge_timeout_seconds)
        ),
        part_refine_after_merge_use_relative_threshold=_boolean(
            pipeline_raw.get(
                "part_refine_after_merge_use_relative_threshold",
                defaults.part_refine_after_merge_use_relative_threshold,
            ),
            name="pipeline.part_refine_after_merge_use_relative_threshold",
        ),
        post_process_use_relative_threshold=_boolean(
            pipeline_raw.get(
                "post_process_use_relative_threshold",
                defaults.post_process_use_relative_threshold,
            ),
            name="pipeline.post_process_use_relative_threshold",
        ),
        cleanup_intermediates=_boolean(
            pipeline_raw.get("cleanup_intermediates", defaults.cleanup_intermediates),
            name="pipeline.cleanup_intermediates",
        ),
        remove_part_glbs_after_render=_boolean(
            pipeline_raw.get(
                "remove_part_glbs_after_render", defaults.remove_part_glbs_after_render
            ),
            name="pipeline.remove_part_glbs_after_render",
        ),
        resume=_boolean(
            pipeline_raw.get("resume", defaults.resume), name="pipeline.resume"
        ),
        adopt_existing_outputs=_boolean(
            pipeline_raw.get("adopt_existing_outputs", defaults.adopt_existing_outputs),
            name="pipeline.adopt_existing_outputs",
        ),
        fail_fast=_boolean(
            pipeline_raw.get("fail_fast", defaults.fail_fast), name="pipeline.fail_fast"
        ),
        workers=int(pipeline_raw.get("workers", defaults.workers)),
        start=int(pipeline_raw.get("start", defaults.start)),
        end=(int(pipeline_raw["end"]) if pipeline_raw.get("end") is not None else None),
    )

    runtime_defaults = RuntimeConfig()
    env_dir = _expand_path(runtime_raw.get("env_dir"))
    runtime = RuntimeConfig(
        blender=str(runtime_raw.get("blender", runtime_defaults.blender)),
        env_dir=env_dir,
        sam_python=_expand_path(runtime_raw.get("sam_python")),
        partfield_python=_expand_path(runtime_raw.get("partfield_python")),
        sam_checkpoint=_expand_path(
            runtime_raw.get("sam_checkpoint", runtime_defaults.sam_checkpoint)
        )
        or runtime_defaults.sam_checkpoint,
        partfield_checkpoint=_expand_path(
            runtime_raw.get(
                "partfield_checkpoint", runtime_defaults.partfield_checkpoint
            )
        )
        or runtime_defaults.partfield_checkpoint,
        partfield_config=_expand_path(
            runtime_raw.get("partfield_config", runtime_defaults.partfield_config)
        )
        or runtime_defaults.partfield_config,
        command_timeout_seconds=(
            int(runtime_raw["command_timeout_seconds"])
            if runtime_raw.get("command_timeout_seconds") is not None
            else None
        ),
        gpu=(
            str(runtime_raw.get("gpu", runtime_defaults.gpu))
            if runtime_raw.get("gpu", runtime_defaults.gpu) is not None
            else None
        ),
        lock_stale_seconds=int(
            runtime_raw.get("lock_stale_seconds", runtime_defaults.lock_stale_seconds)
        ),
    )

    gpt_defaults = GPTConfig()
    gpt = GPTConfig(
        model=str(gpt_raw.get("model", gpt_defaults.model)),
        base_url=(str(gpt_raw["base_url"]) if gpt_raw.get("base_url") else None),
        api_key_env=str(gpt_raw.get("api_key_env", gpt_defaults.api_key_env)),
        basic_prompt=_expand_path(
            gpt_raw.get("basic_prompt", gpt_defaults.basic_prompt)
        )
        or gpt_defaults.basic_prompt,
        axis_prompt=_expand_path(gpt_raw.get("axis_prompt", gpt_defaults.axis_prompt))
        or gpt_defaults.axis_prompt,
    )

    config = AppConfig(
        dataset=dataset,
        pipeline=pipeline,
        runtime=runtime,
        gpt=gpt,
        source=config_path,
    )
    config.validate()
    return config
