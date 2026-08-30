"""Pipeline orchestration, state tracking, locking, and parallel execution."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import AppConfig

LOGGER = logging.getLogger("uniphys.pipeline")
STATE_VERSION = 1
SUCCESS_STATES = {"succeeded", "adopted", "skipped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineError(RuntimeError):
    """Base class for orchestration errors."""


class StageOutputError(PipelineError):
    """Raised when a stage returns without producing its declared outputs."""


class EntityBusyError(PipelineError):
    """Raised when another process already owns an entity lock."""


@dataclass(frozen=True)
class EntityContext:
    """Immutable paths and configuration for one mesh entity."""

    config: AppConfig
    mesh_path: Path

    @property
    def entity(self) -> str:
        return self.mesh_path.stem

    @property
    def type_name(self) -> str:
        return self.config.dataset.type_name

    @property
    def render_entity_dir(self) -> Path:
        return self.config.dataset.render_root / self.type_name / self.entity

    @property
    def view_dir(self) -> Path:
        return self.render_entity_dir / "views"

    @property
    def exr_dir(self) -> Path:
        return self.render_entity_dir / "exrs"

    @property
    def seg_dir(self) -> Path:
        return self.render_entity_dir / "seg"

    @property
    def decomposition_dir(self) -> Path:
        return self.config.dataset.decomposition_root / self.entity

    @property
    def gpt_output_dir(self) -> Path:
        return self.config.dataset.gpt_output_root

    @property
    def metadata_dir(self) -> Path:
        return self.decomposition_dir / ".pipeline"

    @property
    def state_file(self) -> Path:
        return self.metadata_dir / "state.json"


class Stage(ABC):
    """One resumable unit of pipeline work."""

    number: int
    name: str
    description: str
    dependencies: tuple[str, ...] = ()

    def missing_inputs(self, context: EntityContext) -> list[str]:
        return [] if context.mesh_path.exists() else [str(context.mesh_path)]

    @abstractmethod
    def run(self, context: EntityContext) -> None:
        """Execute this stage or raise an exception."""

    @abstractmethod
    def outputs_valid(self, context: EntityContext) -> bool:
        """Return whether the externally visible outputs are complete."""


class StageRegistry:
    """Validated, ordered registry of pipeline stages."""

    def __init__(self, stages: Iterable[Stage] = ()) -> None:
        self._by_name: dict[str, Stage] = {}
        self._by_number: dict[int, Stage] = {}
        for stage in stages:
            self.register(stage)
        self.validate()

    def register(self, stage: Stage) -> None:
        if stage.name in self._by_name:
            raise PipelineError(f"Duplicate stage name: {stage.name}")
        if stage.number in self._by_number:
            raise PipelineError(f"Duplicate stage number: {stage.number}")
        self._by_name[stage.name] = stage
        self._by_number[stage.number] = stage

    def validate(self) -> None:
        for stage in self._by_name.values():
            unknown = set(stage.dependencies) - self._by_name.keys()
            if unknown:
                raise PipelineError(
                    f"Stage '{stage.name}' has unknown dependencies: {sorted(unknown)}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise PipelineError(f"Cycle detected at pipeline stage '{name}'.")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self._by_name[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self._by_name:
            visit(name)

    @property
    def ordered(self) -> tuple[Stage, ...]:
        return tuple(sorted(self._by_name.values(), key=lambda stage: stage.number))

    def resolve(self, token: str | int) -> Stage:
        raw = str(token).strip()
        if raw.isdigit() and int(raw) in self._by_number:
            return self._by_number[int(raw)]
        normalized = raw.lower().replace("-", "_")
        try:
            return self._by_name[normalized]
        except KeyError as exc:
            choices = ", ".join(f"{s.number}:{s.name}" for s in self.ordered)
            raise PipelineError(
                f"Unknown stage '{token}'. Available stages: {choices}"
            ) from exc

    def select(
        self,
        requested: Sequence[str] | None = None,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
        with_dependencies: bool = False,
    ) -> tuple[Stage, ...]:
        if requested and (from_stage or to_stage):
            raise PipelineError(
                "Use either --stages or --from-stage/--to-stage, not both."
            )

        if from_stage or to_stage:
            start_number = (
                self.resolve(from_stage).number
                if from_stage
                else self.ordered[0].number
            )
            end_number = (
                self.resolve(to_stage).number if to_stage else self.ordered[-1].number
            )
            if end_number < start_number:
                raise PipelineError("--to-stage must not precede --from-stage.")
            selected_names = {
                stage.name
                for stage in self.ordered
                if start_number <= stage.number <= end_number
            }
        elif requested:
            selected_names = {self.resolve(token).name for token in requested}
        else:
            selected_names = {stage.name for stage in self.ordered}

        if with_dependencies:
            pending = list(selected_names)
            while pending:
                name = pending.pop()
                for dependency in self._by_name[name].dependencies:
                    if dependency not in selected_names:
                        selected_names.add(dependency)
                        pending.append(dependency)

        return tuple(stage for stage in self.ordered if stage.name in selected_names)


class StateStore:
    """Atomic JSON state store scoped to one entity."""

    def __init__(self, context: EntityContext) -> None:
        self.context = context
        self.path = context.state_file
        self.data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "entity": self.context.entity,
            "mesh_path": str(self.context.mesh_path),
            "updated_at": utc_now(),
            "stages": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"Invalid state file {self.path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
            raise PipelineError(f"Invalid state structure: {self.path}")
        return data

    def stage_record(self, name: str) -> Mapping[str, Any]:
        record = self.data.get("stages", {}).get(name, {})
        return record if isinstance(record, Mapping) else {}

    def update_stage(self, name: str, **values: Any) -> None:
        stages = self.data.setdefault("stages", {})
        current = stages.setdefault(name, {})
        current.update(values)
        self.data["updated_at"] = utc_now()
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        serialized = (
            json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, self.path)


class EntityLock:
    """Small cross-process lock preventing duplicate work for one entity."""

    def __init__(self, context: EntityContext) -> None:
        self.path = context.metadata_dir / "run.lock"
        self.stale_seconds = context.config.runtime.lock_stale_seconds
        self.acquired = False

    def __enter__(self) -> "EntityLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_seconds:
                self.path.unlink(missing_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": utc_now(),
            },
            sort_keys=True,
        ).encode()
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            owner = self.path.read_text(encoding="utf-8", errors="replace")
            raise EntityBusyError(
                f"Entity is already being processed ({owner})."
            ) from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class StageRunResult:
    stage: str
    status: str
    duration_seconds: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class EntityRunResult:
    entity: str
    mesh_path: str
    stages: tuple[StageRunResult, ...] = ()
    lock_error: str | None = None

    @property
    def successful(self) -> bool:
        if self.lock_error:
            return False
        return all(
            result.status in SUCCESS_STATES | {"planned"} for result in self.stages
        )


@dataclass
class RunSummary:
    results: list[EntityRunResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(result.successful for result in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.succeeded


class PipelineRunner:
    """Execute registered stages for one or more mesh entities."""

    def __init__(self, config: AppConfig, registry: StageRegistry) -> None:
        self.config = config
        self.registry = registry

    def discover_meshes(self, entities: Sequence[str] | None = None) -> list[Path]:
        root = self.config.dataset.mesh_root
        if not root.is_dir():
            raise PipelineError(
                f"Mesh root does not exist or is not a directory: {root}"
            )
        extensions = {suffix.lower() for suffix in self.config.dataset.extensions}
        meshes = sorted(
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        duplicate_stems: dict[str, list[Path]] = {}
        for mesh in meshes:
            duplicate_stems.setdefault(mesh.stem, []).append(mesh)
        collisions = {
            name: paths for name, paths in duplicate_stems.items() if len(paths) > 1
        }
        if collisions:
            details = ", ".join(
                f"{name}: {[p.name for p in paths]}"
                for name, paths in collisions.items()
            )
            raise PipelineError(
                f"Mesh filenames must have unique stems; found {details}"
            )

        if entities:
            wanted = {Path(item).stem for item in entities}
            meshes = [mesh for mesh in meshes if mesh.stem in wanted]
            found = {mesh.stem for mesh in meshes}
            missing = sorted(wanted - found)
            if missing:
                raise PipelineError(
                    f"Requested entities were not found: {', '.join(missing)}"
                )

        start = self.config.pipeline.start
        end = self.config.pipeline.end
        return meshes[start:end]

    def run(
        self,
        meshes: Sequence[Path],
        stages: Sequence[Stage],
        *,
        force: Sequence[str] = (),
        dry_run: bool = False,
    ) -> RunSummary:
        stage_names = tuple(stage.name for stage in stages)
        force_names = tuple(self.registry.resolve(name).name for name in force)
        summary = RunSummary()

        if self.config.pipeline.workers == 1 or len(meshes) <= 1:
            for mesh in meshes:
                result = self.run_entity(
                    mesh, stages, force=force_names, dry_run=dry_run
                )
                summary.results.append(result)
                if self.config.pipeline.fail_fast and not result.successful:
                    break
            return summary

        with ProcessPoolExecutor(max_workers=self.config.pipeline.workers) as executor:
            futures = {
                executor.submit(
                    _run_entity_worker,
                    self.config,
                    mesh,
                    stage_names,
                    force_names,
                    dry_run,
                ): mesh
                for mesh in meshes
            }
            for future in as_completed(futures):
                mesh = futures[future]
                try:
                    result = future.result()
                except (
                    Exception
                ) as exc:  # pragma: no cover - defensive process boundary
                    LOGGER.exception("Worker crashed for %s", mesh)
                    result = EntityRunResult(
                        entity=mesh.stem,
                        mesh_path=str(mesh),
                        lock_error=f"Worker crashed: {exc}",
                    )
                summary.results.append(result)
                if self.config.pipeline.fail_fast and not result.successful:
                    for pending in futures:
                        pending.cancel()
                    break
        summary.results.sort(key=lambda result: result.entity)
        return summary

    def run_entity(
        self,
        mesh: Path,
        stages: Sequence[Stage],
        *,
        force: Sequence[str] = (),
        dry_run: bool = False,
    ) -> EntityRunResult:
        context = EntityContext(self.config, mesh)
        results: list[StageRunResult] = []
        force_names = set(force)
        fingerprint = self.config.execution_fingerprint()

        if dry_run:
            return EntityRunResult(
                entity=context.entity,
                mesh_path=str(mesh),
                stages=tuple(StageRunResult(stage.name, "planned") for stage in stages),
            )

        try:
            with EntityLock(context):
                store = StateStore(context)
                current_status: dict[str, str] = {}
                for stage in stages:
                    blocked_dependencies = [
                        dependency
                        for dependency in stage.dependencies
                        if not self._dependency_ready(
                            dependency, context, store, current_status, fingerprint
                        )
                    ]
                    if blocked_dependencies:
                        message = "Dependencies are incomplete: " + ", ".join(
                            blocked_dependencies
                        )
                        result = StageRunResult(stage.name, "blocked", message=message)
                        results.append(result)
                        current_status[stage.name] = result.status
                        continue

                    record = store.stage_record(stage.name)
                    may_resume = (
                        self.config.pipeline.resume
                        and stage.name not in force_names
                        and record.get("status") == "succeeded"
                        and record.get("fingerprint") == fingerprint
                        and stage.outputs_valid(context)
                    )
                    if may_resume:
                        result = StageRunResult(
                            stage.name, "skipped", message="Already complete"
                        )
                        results.append(result)
                        current_status[stage.name] = result.status
                        continue

                    may_adopt = (
                        self.config.pipeline.resume
                        and self.config.pipeline.adopt_existing_outputs
                        and stage.name not in force_names
                        and not record
                        and stage.outputs_valid(context)
                    )
                    if may_adopt:
                        store.update_stage(
                            stage.name,
                            status="succeeded",
                            adopted=True,
                            fingerprint=fingerprint,
                            finished_at=utc_now(),
                            duration_seconds=0.0,
                        )
                        result = StageRunResult(
                            stage.name, "adopted", message="Existing outputs adopted"
                        )
                        results.append(result)
                        current_status[stage.name] = result.status
                        continue

                    missing = stage.missing_inputs(context)
                    if missing:
                        message = "Missing inputs: " + ", ".join(missing)
                        store.update_stage(
                            stage.name,
                            status="failed",
                            fingerprint=fingerprint,
                            finished_at=utc_now(),
                            error=message,
                        )
                        result = StageRunResult(stage.name, "failed", message=message)
                        results.append(result)
                        current_status[stage.name] = result.status
                        if self.config.pipeline.fail_fast:
                            break
                        continue

                    started = time.monotonic()
                    store.update_stage(
                        stage.name,
                        status="running",
                        fingerprint=fingerprint,
                        started_at=utc_now(),
                        error=None,
                    )
                    LOGGER.info("[%s] starting stage %s", context.entity, stage.name)
                    try:
                        stage.run(context)
                        if not stage.outputs_valid(context):
                            raise StageOutputError(
                                f"Stage '{stage.name}' finished but its output validation failed."
                            )
                    except Exception as exc:
                        duration = time.monotonic() - started
                        error_log = self._write_error_log(context, stage, exc)
                        message = f"{type(exc).__name__}: {exc}"
                        store.update_stage(
                            stage.name,
                            status="failed",
                            fingerprint=fingerprint,
                            finished_at=utc_now(),
                            duration_seconds=round(duration, 3),
                            error=message,
                            error_log=str(error_log),
                        )
                        LOGGER.error(
                            "[%s] stage %s failed: %s",
                            context.entity,
                            stage.name,
                            message,
                        )
                        result = StageRunResult(stage.name, "failed", duration, message)
                        results.append(result)
                        current_status[stage.name] = result.status
                        if self.config.pipeline.fail_fast:
                            break
                        continue

                    duration = time.monotonic() - started
                    store.update_stage(
                        stage.name,
                        status="succeeded",
                        fingerprint=fingerprint,
                        adopted=False,
                        finished_at=utc_now(),
                        duration_seconds=round(duration, 3),
                        error=None,
                    )
                    LOGGER.info(
                        "[%s] completed stage %s in %.1fs",
                        context.entity,
                        stage.name,
                        duration,
                    )
                    result = StageRunResult(stage.name, "succeeded", duration)
                    results.append(result)
                    current_status[stage.name] = result.status
        except EntityBusyError as exc:
            return EntityRunResult(
                entity=context.entity,
                mesh_path=str(mesh),
                stages=tuple(results),
                lock_error=str(exc),
            )

        return EntityRunResult(
            entity=context.entity,
            mesh_path=str(mesh),
            stages=tuple(results),
        )

    def _dependency_ready(
        self,
        dependency: str,
        context: EntityContext,
        store: StateStore,
        current_status: Mapping[str, str],
        fingerprint: str,
    ) -> bool:
        if current_status.get(dependency) in SUCCESS_STATES | {"planned"}:
            return True
        stage = self.registry.resolve(dependency)
        record = store.stage_record(dependency)
        if (
            record.get("status") == "succeeded"
            and record.get("fingerprint") == fingerprint
            and stage.outputs_valid(context)
        ):
            return True
        return self.config.pipeline.adopt_existing_outputs and stage.outputs_valid(
            context
        )

    @staticmethod
    def _write_error_log(context: EntityContext, stage: Stage, exc: Exception) -> Path:
        error_dir = context.metadata_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        path = error_dir / f"{stage.number:02d}_{stage.name}.log"
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        path.write_text(detail, encoding="utf-8")
        return path


def _run_entity_worker(
    config: AppConfig,
    mesh: Path,
    stage_names: Sequence[str],
    force_names: Sequence[str],
    dry_run: bool,
) -> EntityRunResult:
    if config.runtime.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.runtime.gpu
    from .stages import create_registry

    registry = create_registry()
    stages = tuple(registry.resolve(name) for name in stage_names)
    return PipelineRunner(config, registry).run_entity(
        mesh,
        stages,
        force=force_names,
        dry_run=dry_run,
    )
