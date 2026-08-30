#!/usr/bin/env python3
"""Convert OBJ/GLB assets into pipeline-ready, single-mesh GLB files.

This utility is deliberately independent from the UniPhys pipeline. It reads
raw assets from one location and writes normalized GLB files to another; the
pipeline should only be pointed at the normalized output directory afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_SUFFIXES = {".glb", ".obj"}


class PreprocessConfigurationError(ValueError):
    """Raised when the requested batch cannot be mapped to unique outputs."""


@dataclass(frozen=True)
class MeshJob:
    source: Path
    output: Path


@dataclass(frozen=True)
class MeshResult:
    source: str
    output: str
    status: str
    unique_mesh_geometries: int | None = None
    mesh_instances: int | None = None
    vertices: int | None = None
    faces: int | None = None
    duration_seconds: float = 0.0
    message: str = ""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_sources(
    input_path: Path,
    *,
    recursive: bool,
    excluded_root: Path | None = None,
) -> list[Path]:
    """Return deterministically ordered OBJ/GLB files from a file or directory."""

    source = input_path.expanduser().resolve()
    if not source.exists():
        raise PreprocessConfigurationError(f"Input does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise PreprocessConfigurationError(
                f"Unsupported input extension '{source.suffix}'; expected {supported}."
            )
        return [source]
    if not source.is_dir():
        raise PreprocessConfigurationError(
            f"Input is neither a regular file nor a directory: {source}"
        )

    excluded = excluded_root.expanduser().resolve() if excluded_root else None
    candidates = source.rglob("*") if recursive else source.iterdir()
    meshes = sorted(
        path.resolve()
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not (excluded is not None and _is_below(path.resolve(), excluded))
    )
    if not meshes:
        scope = "recursively" if recursive else "in the top level"
        raise PreprocessConfigurationError(
            f"No .obj or .glb files were found {scope} of {source}."
        )
    return meshes


def build_jobs(
    sources: Sequence[Path],
    output_dir: Path,
    *,
    name_from: str,
) -> list[MeshJob]:
    """Map source paths to unique, flat GLB output paths."""

    destination = output_dir.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise PreprocessConfigurationError(
            f"Output path exists but is not a directory: {destination}"
        )
    jobs: list[MeshJob] = []
    targets: dict[str, Path] = {}
    for source in sources:
        entity = source.stem if name_from == "stem" else source.parent.name
        if not entity:
            raise PreprocessConfigurationError(
                f"Could not derive an entity name from {source}."
            )
        output = destination / f"{entity}.glb"
        if source.resolve() == output.resolve():
            raise PreprocessConfigurationError(
                f"Input and output are the same file: {source}. "
                "Choose a separate output directory."
            )
        if source.parent.resolve() == destination:
            raise PreprocessConfigurationError(
                f"Output directory contains the source file: {source}. "
                "Keep raw and normalized data in separate directories."
            )
        collision_key = output.name.casefold()
        previous = targets.get(collision_key)
        if previous is not None:
            raise PreprocessConfigurationError(
                "Multiple inputs map to the same output "
                f"'{output.name}': {previous} and {source}. "
                "Use a different input layout or --name-from setting."
            )
        targets[collision_key] = source
        jobs.append(MeshJob(source=source, output=output))
    return jobs


def process_mesh(
    job: MeshJob,
    *,
    overwrite: bool,
    max_faces: int,
) -> MeshResult:
    """Bake scene transforms, concatenate mesh instances, and export one GLB."""

    started = time.monotonic()
    if job.output.exists() and not overwrite:
        return MeshResult(
            source=str(job.source),
            output=str(job.output),
            status="skipped",
            duration_seconds=round(time.monotonic() - started, 3),
            message="Output already exists; use --overwrite to replace it.",
        )

    temporary = job.output.with_name(
        f".{job.output.stem}.{os.getpid()}.tmp.glb"
    )
    try:
        # Keep heavy runtime imports inside the worker so --help and discovery do
        # not require the complete pipeline environment.
        import numpy as np
        import trimesh

        scene = trimesh.load_scene(str(job.source), process=False)
        unique_mesh_geometries = sum(
            isinstance(geometry, trimesh.Trimesh)
            for geometry in scene.geometry.values()
        )

        # Scene.dump() returns a copy of every geometry with its scene-node
        # transform baked in. This also expands repeated geometry instances.
        dumped = list(scene.dump())
        unsupported = [
            type(geometry).__name__
            for geometry in dumped
            if not isinstance(geometry, trimesh.Trimesh)
        ]
        if unsupported:
            kinds = ", ".join(sorted(set(unsupported)))
            raise ValueError(f"contains unsupported non-mesh geometry: {kinds}")
        mesh_instances = [
            geometry for geometry in dumped if len(geometry.faces) > 0
        ]
        if not mesh_instances:
            raise ValueError("contains no triangle mesh faces")

        mesh = (
            mesh_instances[0].copy()
            if len(mesh_instances) == 1
            else trimesh.util.concatenate(mesh_instances)
        )
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(
                f"concatenation returned {type(mesh).__name__}, not Trimesh"
            )
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            raise ValueError(
                f"expected triangular faces, got shape {tuple(mesh.faces.shape)}"
            )
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("normalized mesh is empty")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("mesh vertices contain NaN or infinite coordinates")
        if max_faces and len(mesh.faces) > max_faces:
            raise ValueError(
                f"mesh has {len(mesh.faces):,} faces, exceeding "
                f"--max-faces={max_faces:,}"
            )

        job.output.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        mesh.export(str(temporary), file_type="glb")

        # Verify that the output itself contains exactly one mesh geometry and
        # one mesh instance before checking the downstream mesh-coercion path.
        verified_scene = trimesh.load_scene(str(temporary), process=False)
        verified_geometries = [
            geometry
            for geometry in verified_scene.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        verified_instances = [
            geometry
            for geometry in verified_scene.dump()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if len(verified_geometries) != 1 or len(verified_instances) != 1:
            raise ValueError(
                "exported GLB is not a single mesh: "
                f"{len(verified_geometries)} mesh geometries, "
                f"{len(verified_instances)} mesh instances"
            )

        # Reload through the same mesh-coercion path used by downstream
        # geometry code. A count mismatch indicates an invalid round trip.
        verified = trimesh.load(
            str(temporary), force="mesh", process=False
        )
        if not isinstance(verified, trimesh.Trimesh):
            raise TypeError(
                "exported GLB could not be reloaded as a Trimesh object"
            )
        if len(verified.faces) != len(mesh.faces):
            raise ValueError(
                "face count changed during GLB round trip: "
                f"{len(mesh.faces):,} -> {len(verified.faces):,}"
            )
        if not np.isfinite(verified.vertices).all():
            raise ValueError(
                "exported GLB contains NaN or infinite vertex coordinates"
            )

        os.replace(temporary, job.output)
        return MeshResult(
            source=str(job.source),
            output=str(job.output),
            status="processed",
            unique_mesh_geometries=unique_mesh_geometries,
            mesh_instances=len(mesh_instances),
            vertices=len(verified.vertices),
            faces=len(verified.faces),
            duration_seconds=round(time.monotonic() - started, 3),
        )
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return MeshResult(
            source=str(job.source),
            output=str(job.output),
            status="failed",
            duration_seconds=round(time.monotonic() - started, 3),
            message=f"{type(exc).__name__}: {exc}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OBJ/GLB files into separate, pipeline-ready GLB files by "
            "baking scene transforms and concatenating all mesh instances."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="An OBJ/GLB file or a directory containing mesh files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Separate directory in which normalized GLB files will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively instead of only their top level.",
    )
    parser.add_argument(
        "--name-from",
        choices=("stem", "parent"),
        default="stem",
        help=(
            "Derive output names from the source filename stem or its parent "
            "directory (default: stem)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Number of mesh conversion processes (default: 1).",
    )
    parser.add_argument(
        "--max-faces",
        type=_nonnegative_int,
        default=500_000,
        help="Reject meshes above this face count; 0 disables the limit (default: 500000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output GLBs. Source files are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the source-to-output mapping without loading or writing meshes.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: OUTPUT_DIR/preprocess_report.json).",
    )
    return parser


def _print_result(index: int, total: int, result: MeshResult) -> None:
    prefix = {
        "processed": "OK",
        "skipped": "SKIP",
        "failed": "FAIL",
    }.get(result.status, result.status.upper())
    detail = ""
    if result.status == "processed":
        detail = (
            f" ({result.mesh_instances} source mesh instance(s), "
            f"{result.faces:,} faces)"
        )
    elif result.message:
        detail = f" ({result.message})"
    print(
        f"[{index}/{total}] {prefix}: {result.source} -> {result.output}{detail}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        sources = discover_sources(
            input_path,
            recursive=args.recursive,
            excluded_root=output_dir if _is_below(output_dir, input_path) else None,
        )
        jobs = build_jobs(sources, output_dir, name_from=args.name_from)
    except PreprocessConfigurationError as exc:
        parser.error(str(exc))

    print(f"Input meshes: {len(jobs)}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] PLAN: {job.source} -> {job.output}")
        return 0

    results: list[MeshResult] = []
    if args.workers == 1 or len(jobs) == 1:
        for index, job in enumerate(jobs, start=1):
            result = process_mesh(
                job,
                overwrite=args.overwrite,
                max_faces=args.max_faces,
            )
            results.append(result)
            _print_result(index, len(jobs), result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_mesh,
                    job,
                    overwrite=args.overwrite,
                    max_faces=args.max_faces,
                ): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _print_result(index, len(jobs), result)

    results.sort(key=lambda item: item.source)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output_dir / "preprocess_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total": len(results),
        "processed": sum(item.status == "processed" for item in results),
        "skipped": sum(item.status == "skipped" for item in results),
        "failed": sum(item.status == "failed" for item in results),
        "results": [asdict(item) for item in results],
    }
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Summary: "
        f"processed={summary['processed']} skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    print(f"Report: {report_path}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
