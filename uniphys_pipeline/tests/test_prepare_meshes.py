from pathlib import Path

import pytest

from pre_process.prepare_meshes import (
    PreprocessConfigurationError,
    build_jobs,
    discover_sources,
)


def test_discover_sources_filters_extensions_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.obj").touch()
    (tmp_path / "a.GLB").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.obj").touch()

    flat = discover_sources(tmp_path, recursive=False)
    recursive = discover_sources(tmp_path, recursive=True)

    assert [path.name for path in flat] == ["a.GLB", "b.obj"]
    assert [path.name for path in recursive] == ["a.GLB", "b.obj", "c.obj"]


def test_discover_sources_excludes_nested_output_directory(tmp_path: Path) -> None:
    (tmp_path / "source.obj").touch()
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "old.glb").touch()

    sources = discover_sources(
        tmp_path,
        recursive=True,
        excluded_root=output,
    )

    assert [path.name for path in sources] == ["source.obj"]


def test_build_jobs_can_name_nested_raw_models_from_parent(tmp_path: Path) -> None:
    source = tmp_path / "entity-001" / "raw_model.obj"
    source.parent.mkdir()
    source.touch()
    output = tmp_path / "prepared"

    jobs = build_jobs([source.resolve()], output, name_from="parent")

    assert jobs[0].output == output.resolve() / "entity-001.glb"


def test_build_jobs_rejects_duplicate_output_names(tmp_path: Path) -> None:
    first = tmp_path / "a" / "model.obj"
    second = tmp_path / "b" / "model.glb"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(PreprocessConfigurationError, match="same output"):
        build_jobs(
            [first.resolve(), second.resolve()],
            tmp_path / "prepared",
            name_from="stem",
        )


def test_build_jobs_requires_separate_source_and_output_directories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.obj"
    source.touch()

    with pytest.raises(PreprocessConfigurationError, match="separate directories"):
        build_jobs([source.resolve()], tmp_path, name_from="stem")
