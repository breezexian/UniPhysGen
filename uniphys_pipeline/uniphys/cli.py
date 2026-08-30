"""Command-line interface for the refactored UniPhys pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import PROJECT_ROOT, AppConfig, ConfigError, load_config
from .core import EntityContext, PipelineError, PipelineRunner
from .stages import create_registry

LOGGER = logging.getLogger("uniphys")
_REPOSITORY_CONFIG = PROJECT_ROOT / "configs/example.yaml"
DEFAULT_CONFIG = _REPOSITORY_CONFIG if _REPOSITORY_CONFIG.is_file() else None


def _parse_blender_version(output: str) -> tuple[int, int, int] | None:
    """Extract a semantic Blender version from ``blender --version`` output."""

    match = re.search(r"\bBlender\s+(\d+)\.(\d+)\.(\d+)\b", output)
    if match is None:
        return None
    major, minor, patch = (int(value) for value in match.groups())
    return major, minor, patch


def _detect_blender_version(executable: str) -> tuple[int, int, int] | None:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_blender_version(f"{result.stdout}\n{result.stderr}")


def _split_values(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(items)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    default_description = (
        str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT))
        if DEFAULT_CONFIG is not None
        else "built-in defaults"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML or JSON configuration file (default: {default_description})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Run and inspect the UniPhys 3D physical annotation pipeline.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run pipeline stages for mesh entities.")
    _add_config_argument(run)
    run.add_argument(
        "--stages",
        nargs="+",
        help="Stage numbers/names, separated by spaces or commas. Defaults to config stages.",
    )
    run.add_argument("--from-stage", help="Run from this stage, inclusive.")
    run.add_argument("--to-stage", help="Run through this stage, inclusive.")
    run.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Automatically include prerequisites of explicitly selected stages.",
    )
    run.add_argument(
        "--entity",
        action="append",
        help="Only run a mesh filename or stem; repeat for multiple entities.",
    )
    run.add_argument(
        "--start", type=int, help="Start index after deterministic sorting."
    )
    run.add_argument("--end", type=int, help="Exclusive end index after sorting.")
    run.add_argument("--workers", type=int, help="Number of entity worker processes.")
    run.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume completed stages (default comes from config).",
    )
    run.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop scheduling after the first failed entity.",
    )
    run.add_argument(
        "--force", action="append", help="Force a stage to rerun; repeatable."
    )
    run.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for worker processes (default: 0).",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing files."
    )

    status = subparsers.add_parser("status", help="Show per-entity stage state.")
    _add_config_argument(status)
    status.add_argument(
        "--entity", action="append", help="Filter by mesh stem or filename."
    )
    status.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )

    doctor = subparsers.add_parser(
        "doctor", help="Check paths, executables, models, and secrets."
    )
    _add_config_argument(doctor)

    subparsers.add_parser(
        "list-stages", help="List canonical pipeline stages and dependencies."
    )
    return parser


def _load_and_override(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if args.command != "run":
        return config
    stages = _split_values(args.stages)
    return config.with_overrides(
        stages=stages,
        workers=args.workers,
        start=args.start,
        end=args.end,
        resume=args.resume,
        fail_fast=args.fail_fast,
        gpu=args.gpu,
    )


def _run(args: argparse.Namespace) -> int:
    config = _load_and_override(args)
    if config.runtime.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.runtime.gpu
    registry = create_registry()
    requested = None if (args.from_stage or args.to_stage) else config.pipeline.stages
    stages = registry.select(
        requested,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        with_dependencies=args.with_dependencies,
    )
    force = _split_values(args.force) or ()
    runner = PipelineRunner(config, registry)
    meshes = runner.discover_meshes(args.entity)
    if not meshes:
        raise PipelineError("No supported mesh files matched the current selection.")

    print(f"Dataset: {config.dataset.name}/{config.dataset.type_name}")
    print(f"Meshes: {len(meshes)}")
    print("Stages: " + " -> ".join(f"{stage.number}:{stage.name}" for stage in stages))
    if args.dry_run:
        print("Mode: dry-run (no files will be written)")

    summary = runner.run(meshes, stages, force=force, dry_run=args.dry_run)
    for entity_result in summary.results:
        if entity_result.lock_error:
            print(f"[{entity_result.entity}] busy: {entity_result.lock_error}")
            continue
        status_text = ", ".join(
            f"{stage.stage}={stage.status}" for stage in entity_result.stages
        )
        print(f"[{entity_result.entity}] {status_text}")
    print(
        f"Summary: total={len(summary.results)} succeeded={summary.succeeded} failed={summary.failed}"
    )
    return 0 if summary.failed == 0 else 1


def _status(args: argparse.Namespace) -> int:
    config = _load_and_override(args)
    registry = create_registry()
    runner = PipelineRunner(config, registry)
    meshes = runner.discover_meshes(args.entity)
    records: list[dict[str, object]] = []
    for mesh in meshes:
        context = EntityContext(config, mesh)
        if context.state_file.exists():
            try:
                state = json.loads(context.state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                state = {"error": str(exc), "stages": {}}
        else:
            state = {"stages": {}}
        stages = state.get("stages", {}) if isinstance(state, dict) else {}
        stage_status = {
            stage.name: (
                stages.get(stage.name, {}).get("status", "pending")
                if isinstance(stages, dict)
                and isinstance(stages.get(stage.name, {}), dict)
                else "pending"
            )
            for stage in registry.ordered
        }
        records.append(
            {
                "entity": context.entity,
                "mesh_path": str(mesh),
                "state_file": str(context.state_file),
                "stages": stage_status,
                **(
                    {"error": state["error"]}
                    if isinstance(state, dict) and "error" in state
                    else {}
                ),
            }
        )
    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return 0
    for record in records:
        statuses = record["stages"]
        assert isinstance(statuses, dict)
        completed = sum(
            status in {"succeeded", "adopted"} for status in statuses.values()
        )
        failed = [name for name, status in statuses.items() if status == "failed"]
        detail = f"{completed}/{len(registry.ordered)} complete"
        if failed:
            detail += f"; failed={','.join(failed)}"
        print(f"[{record['entity']}] {detail}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = _load_and_override(args)
    registry = create_registry()
    requested_stages = {stage.name for stage in registry.select(config.pipeline.stages)}
    checks: list[tuple[str, str, str]] = []

    def check(label: str, ok: bool, detail: str, *, warning: bool = False) -> None:
        status = "OK" if ok else ("WARN" if warning else "ERROR")
        checks.append((status, label, detail))

    check("Python", sys.version_info >= (3, 10), sys.version.split()[0])
    check(
        "Mesh root",
        config.dataset.mesh_root.is_dir(),
        str(config.dataset.mesh_root),
    )
    blender = shutil.which(config.runtime.blender) or (
        config.runtime.blender if Path(config.runtime.blender).exists() else None
    )
    check("Blender", blender is not None, str(blender or config.runtime.blender))
    if blender is not None:
        blender_version = _detect_blender_version(str(blender))
        version_text = (
            ".".join(str(value) for value in blender_version)
            if blender_version is not None
            else "could not parse `blender --version`"
        )
        if blender_version == (4, 5, 3):
            check("Blender version", True, f"{version_text} (recommended)")
        elif blender_version is not None and blender_version[:2] == (4, 5):
            check(
                "Blender version",
                False,
                f"{version_text}; Blender 4.5.3 is recommended",
                warning=True,
            )
        else:
            check(
                "Blender version",
                False,
                f"{version_text}; Blender 4.5.x is required (4.5.3 recommended)",
            )

    module_requirements = {
        "render": {"trimesh"},
        "decompose": {"OpenEXR", "mlxtend", "numpy", "scipy", "trimesh"},
        "export_parts": {"PIL", "cv2", "numpy", "trimesh"},
        "annotate_basic": {"openai"},
        "build_kinematic_graph": {"matplotlib", "networkx", "trimesh"},
        "propose_kinematics": {"open3d", "scipy", "torch", "trimesh"},
        "render_axes": {"cv2", "matplotlib"},
        "annotate_revolute": {"openai"},
        "annotate_prismatic": {"openai"},
        "generate_mujoco": {"numpy", "trimesh"},
        "validate_basic": {"numpy", "trimesh"},
        "validate_simulation": {"mujoco", "open3d", "torch"},
    }
    required_modules = sorted(
        {
            module
            for stage_name in requested_stages
            for module in module_requirements.get(stage_name, set())
        }
    )
    for module in required_modules:
        check(f"Python module {module}", find_spec(module) is not None, sys.executable)

    if "decompose" in requested_stages:
        check(
            "SAM Python",
            config.runtime.resolved_sam_python.is_file(),
            str(config.runtime.resolved_sam_python),
        )
        check(
            "SAM checkpoint",
            config.runtime.sam_checkpoint.is_file(),
            str(config.runtime.sam_checkpoint),
        )
        check(
            "PartField Python",
            config.runtime.resolved_partfield_python.is_file(),
            str(config.runtime.resolved_partfield_python),
        )
        check(
            "PartField checkpoint",
            config.runtime.partfield_checkpoint.is_file(),
            str(config.runtime.partfield_checkpoint),
        )

    gpt_stages = {"annotate_basic", "annotate_revolute", "annotate_prismatic"}
    if requested_stages & gpt_stages:
        check(
            config.gpt.api_key_env,
            bool(os.environ.get(config.gpt.api_key_env)),
            "configured" if os.environ.get(config.gpt.api_key_env) else "not set",
        )
        check(
            "Basic prompt",
            config.gpt.basic_prompt.is_file(),
            str(config.gpt.basic_prompt),
        )
        check(
            "Axis prompt", config.gpt.axis_prompt.is_file(), str(config.gpt.axis_prompt)
        )

    for status, label, detail in checks:
        print(f"[{status:5}] {label}: {detail}")
    errors = sum(status == "ERROR" for status, _, _ in checks)
    print(f"Doctor summary: {len(checks) - errors}/{len(checks)} checks passed")
    return 0 if errors == 0 else 1


def _list_stages() -> int:
    registry = create_registry()
    for stage in registry.ordered:
        dependencies = ", ".join(stage.dependencies) if stage.dependencies else "-"
        print(f"{stage.number:2d}  {stage.name:24} deps={dependencies}")
        print(f"    {stage.description}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "status":
            return _status(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "list-stages":
            return _list_stages()
    except (ConfigError, PipelineError) as exc:
        LOGGER.error("%s", exc)
        return 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
