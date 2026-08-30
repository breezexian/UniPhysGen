from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from uniphys.config import AppConfig, DatasetConfig, PipelineConfig
from uniphys.core import EntityContext, PipelineRunner, Stage, StageRegistry


class TouchStage(Stage):
    number = 1
    name = "touch"
    description = "Create a marker."

    def run(self, context: EntityContext) -> None:
        context.decomposition_dir.mkdir(parents=True, exist_ok=True)
        (context.decomposition_dir / "touch.done").write_text("ok\n", encoding="utf-8")

    def outputs_valid(self, context: EntityContext) -> bool:
        return (context.decomposition_dir / "touch.done").is_file()


class DependentStage(Stage):
    number = 2
    name = "dependent"
    description = "Depend on touch."
    dependencies = ("touch",)

    def run(self, context: EntityContext) -> None:
        context.decomposition_dir.mkdir(parents=True, exist_ok=True)
        (context.decomposition_dir / "dependent.done").write_text(
            "ok\n", encoding="utf-8"
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        return (context.decomposition_dir / "dependent.done").is_file()


def make_config(tmp_path: Path) -> AppConfig:
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    dataset = DatasetConfig(
        name="demo",
        type_name="type",
        input_root=tmp_path,
        output_root=tmp_path / "outputs",
        mesh_root_override=mesh_root,
    )
    pipeline = PipelineConfig(stages=("touch",), workers=1)
    return AppConfig(dataset=dataset, pipeline=pipeline)


def test_registry_selection_and_dependency_expansion() -> None:
    registry = StageRegistry((TouchStage(), DependentStage()))

    selected = registry.select(("dependent",), with_dependencies=True)

    assert [stage.name for stage in selected] == ["touch", "dependent"]
    assert registry.resolve("1").name == "touch"
    assert registry.resolve("dependent").number == 2


def test_runner_writes_state_and_resumes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mesh = config.dataset.mesh_root / "sample.glb"
    mesh.write_bytes(b"fixture")
    registry = StageRegistry((TouchStage(),))
    runner = PipelineRunner(config, registry)

    first = runner.run([mesh], registry.ordered)
    second = runner.run([mesh], registry.ordered)

    assert first.failed == 0
    assert first.results[0].stages[0].status == "succeeded"
    assert second.results[0].stages[0].status == "skipped"
    state_file = EntityContext(config, mesh).state_file
    assert state_file.is_file()


def test_dry_run_does_not_create_output_directories(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mesh = config.dataset.mesh_root / "sample.glb"
    mesh.write_bytes(b"fixture")
    registry = StageRegistry((TouchStage(),))
    runner = PipelineRunner(config, registry)

    result = runner.run([mesh], registry.ordered, dry_run=True)

    assert result.failed == 0
    assert result.results[0].stages[0].status == "planned"
    assert not config.dataset.decomposition_root.exists()


def test_start_end_slicing_is_applied(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for name in ("a.glb", "b.glb", "c.glb"):
        (config.dataset.mesh_root / name).write_bytes(b"fixture")
    config = replace(config, pipeline=replace(config.pipeline, start=1, end=2))
    runner = PipelineRunner(config, StageRegistry((TouchStage(),)))

    meshes = runner.discover_meshes()

    assert [mesh.name for mesh in meshes] == ["b.glb"]
