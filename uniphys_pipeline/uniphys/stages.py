"""Concrete UniPhys pipeline stages built on the existing algorithm modules."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

from .config import PROJECT_ROOT
from .core import EntityContext, PipelineError, Stage, StageRegistry

LOGGER = logging.getLogger("uniphys.stages")


class StageTimeoutError(TimeoutError):
    """Raised when an in-process legacy algorithm exceeds its time limit."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise StageTimeoutError("Stage operation timed out.")


@contextmanager
def _alarm_timeout(seconds: int) -> Iterator[None]:
    """Use SIGALRM where supported; keep the code portable elsewhere."""

    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _run_command(
    context: EntityContext,
    command: Sequence[str | Path],
    *,
    cwd: Path = PROJECT_ROOT,
) -> None:
    args = [str(item) for item in command]
    LOGGER.info("[%s] $ %s", context.entity, shlex.join(args))
    environment = os.environ.copy()
    if context.config.runtime.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = context.config.runtime.gpu
    environment["UNIPHYS_BLENDER"] = context.config.runtime.blender
    subprocess.run(
        args,
        cwd=cwd,
        env=environment,
        check=True,
        timeout=context.config.runtime.command_timeout_seconds,
    )


def _safe_rmtree(path: Path, *, allowed_roots: Sequence[Path]) -> None:
    if not path.exists():
        return
    target = path.resolve()
    roots = [root.resolve() for root in allowed_roots]
    if target in roots or not any(target.is_relative_to(root) for root in roots):
        raise PipelineError(f"Refusing to remove unsafe path: {target}")
    shutil.rmtree(target)


def _json_file_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return True


def _legacy_args(context: EntityContext) -> SimpleNamespace:
    options = context.config.pipeline
    return SimpleNamespace(
        mesh_path=str(context.mesh_path),
        view_dir=str(context.view_dir),
        exr_dir=str(context.exr_dir),
        seg_dir=str(context.seg_dir),
        save_dir=str(context.decomposition_dir),
        render_num=options.render_num,
        show_part=options.show_part,
        max_merge_num=options.max_merge_num,
        cmp_iou_type=options.cmp_iou_type,
        refine_sam_masks=options.refine_sam_masks,
    )


def _expected_motion_parts(context: EntityContext, motion_code: str) -> list[str]:
    path = context.gpt_output_dir / "kg_res" / context.entity / "kinematic_info.json"
    if not _json_file_valid(path):
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    return sorted(
        str(label)
        for label, motions in data.items()
        if isinstance(motions, dict)
        and motion_code in motions
        and isinstance(motions[motion_code], dict)
        and motions[motion_code].get("parent") is not None
    )


class RenderStage(Stage):
    number = 1
    name = "render"
    description = "Render RGB views and face-ID EXR maps with Blender."

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing = super().missing_inputs(context)
        if (
            shutil.which(context.config.runtime.blender) is None
            and not Path(context.config.runtime.blender).exists()
        ):
            missing.append(f"Blender executable: {context.config.runtime.blender}")
        return missing

    def run(self, context: EntityContext) -> None:
        from utils.mesh_handle import read_mesh

        options = context.config.pipeline
        context.view_dir.mkdir(parents=True, exist_ok=True)
        context.exr_dir.mkdir(parents=True, exist_ok=True)

        mesh = read_mesh(str(context.mesh_path))
        num_faces = len(mesh.faces)
        if num_faces > options.max_faces:
            raise PipelineError(
                f"Mesh has {num_faces:,} faces, exceeding max_faces={options.max_faces:,}."
            )

        expected_exr = context.exr_dir / f"view_{options.render_num - 1}_faceID0001.exr"
        if not expected_exr.is_file() or expected_exr.stat().st_size == 0:
            if num_faces < options.faces_per_batch:
                _run_command(
                    context,
                    [
                        context.config.runtime.blender,
                        "--background",
                        "--python",
                        PROJECT_ROOT / "utils/blender/blender_face.py",
                        "--log-level",
                        "0",
                        "--quiet",
                        "--",
                        context.mesh_path,
                        context.exr_dir,
                    ],
                )
            else:
                _run_command(
                    context,
                    [
                        context.config.runtime.blender,
                        "--background",
                        "--python",
                        PROJECT_ROOT / "utils/blender/blender_face_batch.py",
                        "--log-level",
                        "0",
                        "--quiet",
                        "--",
                        context.mesh_path,
                        context.exr_dir,
                        str(num_faces),
                        str(options.faces_per_batch),
                    ],
                )
                from utils.exr_combine import exr_combine_main

                exr_combine_main(
                    str(context.exr_dir),
                    num_faces,
                    options.faces_per_batch,
                    options.render_num,
                )

        expected_view = context.view_dir / f"view_{options.render_num - 1}.png"
        if not expected_view.is_file() or expected_view.stat().st_size == 0:
            _run_command(
                context,
                [
                    context.config.runtime.blender,
                    "--background",
                    "--python",
                    PROJECT_ROOT / "utils/blender/blender_ori.py",
                    "--log-level",
                    "0",
                    "--quiet",
                    "--",
                    context.mesh_path,
                    context.view_dir,
                ],
            )

    def outputs_valid(self, context: EntityContext) -> bool:
        last = context.config.pipeline.render_num - 1
        expected = (
            context.view_dir / f"view_{last}.png",
            context.exr_dir / f"view_{last}_faceID0001.exr",
        )
        return all(path.is_file() and path.stat().st_size > 0 for path in expected)


class DecomposeStage(Stage):
    number = 2
    name = "decompose"
    description = "Run SAM2 projection, PartField matching, refinement, and merging."
    dependencies = ("render",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing = super().missing_inputs(context)
        if not RenderStage().outputs_valid(context):
            missing.append("render stage outputs")
        runtime = context.config.runtime
        if (
            context.config.pipeline.run_sam2_if_missing
            and not self._segmentation_complete(context)
        ):
            if not runtime.resolved_sam_python.is_file():
                missing.append(str(runtime.resolved_sam_python))
            if not runtime.sam_checkpoint.is_file():
                missing.append(str(runtime.sam_checkpoint))
        if not runtime.resolved_partfield_python.is_file():
            missing.append(str(runtime.resolved_partfield_python))
        for path in (runtime.partfield_checkpoint, runtime.partfield_config):
            if not path.is_file():
                missing.append(str(path))
        return missing

    def run(self, context: EntityContext) -> None:
        options = context.config.pipeline
        context.decomposition_dir.mkdir(parents=True, exist_ok=True)
        if not self._segmentation_complete(context):
            if not options.run_sam2_if_missing:
                raise PipelineError(
                    f"SAM2 masks are missing in {context.seg_dir}; enable run_sam2_if_missing."
                )
            _run_command(
                context,
                [
                    context.config.runtime.resolved_sam_python,
                    PROJECT_ROOT / "utils/sam2/sam2.py",
                    "--entity_dir",
                    context.render_entity_dir,
                    "--checkpoint",
                    context.config.runtime.sam_checkpoint,
                ],
                cwd=PROJECT_ROOT / "utils/sam2",
            )
        if not self._segmentation_complete(context):
            raise PipelineError(
                "SAM2 completed without producing all expected segmentation maps."
            )

        from utils.sam_3d_mask import sam_3d_mask_main

        legacy_args = _legacy_args(context)
        sam_masks, multi_view_masks = sam_3d_mask_main(legacy_args)
        init_cluster_num = len(sam_masks)
        if init_cluster_num < 1:
            raise PipelineError("The projected 3D SAM mask is empty.")
        LOGGER.info("[%s] projected %d SAM masks", context.entity, init_cluster_num)

        workspace, feature_dir, cluster_dir = self._partfield_paths(context)
        succeeded = False
        try:
            try:
                self._run_partfield(context, init_cluster_num, use_small_points=False)
            except Exception as first_error:
                LOGGER.warning(
                    "[%s] PartField primary inference failed; retrying with fewer points: %s",
                    context.entity,
                    first_error,
                )
                self._clear_partfield_outputs(context)
                try:
                    self._run_partfield(
                        context, init_cluster_num, use_small_points=True
                    )
                except Exception as retry_error:
                    marker = context.decomposition_dir / "partfield_error.txt"
                    marker.write_text(
                        f"primary: {first_error}\nretry: {retry_error}\n",
                        encoding="utf-8",
                    )
                    raise PipelineError(
                        "PartField failed in both inference modes."
                    ) from retry_error

            best_cluster_file = self._hungarian_match(
                context,
                init_cluster_num,
                multi_view_masks,
            )
            self._refine_and_merge(
                context,
                best_cluster_file,
                multi_view_masks,
            )
            (context.decomposition_dir / "partfield_error.txt").unlink(missing_ok=True)
            succeeded = True
        finally:
            if succeeded and options.cleanup_intermediates:
                allowed = (PROJECT_ROOT, context.decomposition_dir)
                for path in (workspace, feature_dir, cluster_dir):
                    _safe_rmtree(path, allowed_roots=allowed)

    def outputs_valid(self, context: EntityContext) -> bool:
        return (
            _json_file_valid(context.decomposition_dir / "final_post_face2cls.json")
            and (context.decomposition_dir / "record.txt").is_file()
        )

    @staticmethod
    def _segmentation_complete(context: EntityContext) -> bool:
        return all(
            (context.seg_dir / f"view_{index}_s.npy").is_file()
            for index in range(context.config.pipeline.render_num)
        )

    @staticmethod
    def _partfield_paths(context: EntityContext) -> tuple[Path, Path, Path]:
        unique_name = f"{context.type_name}_{context.entity}"
        workspace = context.metadata_dir / "intermediate" / "partfield"
        feature_dir = PROJECT_ROOT / "exp_results" / "partfield_features" / unique_name
        cluster_dir = PROJECT_ROOT / "exp_results" / "clustering" / unique_name
        return workspace, feature_dir, cluster_dir

    def _clear_partfield_outputs(self, context: EntityContext) -> None:
        workspace, feature_dir, cluster_dir = self._partfield_paths(context)
        allowed = (PROJECT_ROOT, context.decomposition_dir)
        for path in (workspace, feature_dir, cluster_dir):
            _safe_rmtree(path, allowed_roots=allowed)

    def _run_partfield(
        self,
        context: EntityContext,
        init_cluster_num: int,
        *,
        use_small_points: bool,
    ) -> None:
        runtime = context.config.runtime
        options = context.config.pipeline
        workspace, feature_dir, cluster_dir = self._partfield_paths(context)
        unique_name = f"{context.type_name}_{context.entity}"
        data_dir = workspace / "data" / unique_name
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(context.mesh_path, data_dir / context.mesh_path.name)

        inference_script = PROJECT_ROOT / "utils/partfield/partfield_inference.py"
        inference_command: list[str | Path] = [
            runtime.resolved_partfield_python,
            inference_script,
            "-c",
            runtime.partfield_config,
            "--opts",
            "continue_ckpt",
            runtime.partfield_checkpoint,
            "result_name",
            f"partfield_features/{unique_name}",
            "dataset.data_path",
            data_dir,
        ]
        if use_small_points:
            inference_command.extend(["n_point_per_face", "500"])
        _run_command(context, inference_command)

        feature_file = feature_dir / f"part_feat_{context.entity}_0_batch.npy"
        if not feature_file.is_file():
            raise PipelineError(
                f"PartField feature file was not produced: {feature_file}"
            )

        cluster_script = (
            PROJECT_ROOT / "utils/partfield/run_part_clustering_candidate.py"
        )
        _run_command(
            context,
            [
                runtime.resolved_partfield_python,
                cluster_script,
                "--root",
                feature_dir,
                "--dump_dir",
                cluster_dir,
                "--source_dir",
                data_dir,
                "--use_agglo",
                "True",
                "--init_num_clusters",
                str(init_cluster_num),
                "--search_scope",
                str(max(init_cluster_num - 1, 1)),
                str(options.cluster_search_max),
                "--option",
                "0",
            ],
        )
        if not (cluster_dir / "cluster_out").is_dir():
            raise PipelineError(
                f"PartField clustering output was not produced: {cluster_dir}"
            )

    def _hungarian_match(
        self,
        context: EntityContext,
        init_cluster_num: int,
        multi_view_masks: Any,
    ) -> Path:
        from utils.hungarian_match import hm_main

        options = context.config.pipeline
        sam_file = (
            context.decomposition_dir / f"sam_face2cls_{options.cmp_iou_type}.json"
        )
        if not _json_file_valid(sam_file):
            raise PipelineError(
                f"3D SAM face mapping is missing or invalid: {sam_file}"
            )
        _, _, cluster_dir = self._partfield_paths(context)
        best_file, _loss = hm_main(
            str(context.mesh_path),
            str(sam_file),
            str(cluster_dir / "cluster_out"),
            context.entity,
            init_cluster_num,
            (options.cluster_search_min, options.cluster_search_max),
            multi_view_masks=multi_view_masks,
            refine=options.refine_sam_masks,
        )
        source = Path(best_file)
        if not source.is_file():
            raise PipelineError(f"Hungarian matching output is missing: {source}")
        destination = context.decomposition_dir / source.name
        shutil.copy2(source, destination)
        return destination

    def _refine_and_merge(
        self,
        context: EntityContext,
        best_cluster_file: Path,
        multi_view_masks: Any,
    ) -> None:
        from utils.apriori_merge_parts import apriori_merge_parts_main
        from utils.merge_part import merge_parts_main
        from utils.part_refine import part_refine_main
        from utils.part_refine_after_merge import part_refine_after_merge_main
        from utils.post_process import post_process_main

        options = context.config.pipeline
        legacy_args = _legacy_args(context)
        with _alarm_timeout(options.part_refine_timeout_seconds):
            new_parts = part_refine_main(
                str(context.mesh_path),
                str(best_cluster_file),
                save_dir=str(context.decomposition_dir),
            )
        if not new_parts:
            raise PipelineError("Part refinement produced no parts.")

        sam_file = (
            context.decomposition_dir / f"sam_face2cls_{options.cmp_iou_type}.json"
        )
        face_to_class = json.loads(sam_file.read_text(encoding="utf-8"))
        mask_to_faces: dict[Any, list[int]] = {}
        for face, class_id in face_to_class.items():
            mask_to_faces.setdefault(class_id, []).append(int(face))
        sam_masks = [set(faces) for faces in mask_to_faces.values()]

        merge_flag: str
        try:
            with _alarm_timeout(options.merge_timeout_seconds):
                if len(new_parts) > 100:
                    merge_flag = (
                        f"normal merge selected for {len(new_parts)} refined parts"
                    )
                    merged_parts = merge_parts_main(
                        str(context.mesh_path),
                        sam_masks,
                        new_parts,
                        save_dir=str(context.decomposition_dir),
                    )
                else:
                    merge_flag = "apriori merge"
                    merged_parts = apriori_merge_parts_main(
                        legacy_args,
                        multi_view_masks,
                        new_parts,
                        str(context.decomposition_dir),
                    )
        except StageTimeoutError:
            merge_flag = "apriori merge timed out; used normal merge"
            merged_parts = merge_parts_main(
                str(context.mesh_path),
                sam_masks,
                new_parts,
                save_dir=str(context.decomposition_dir),
            )
        if len(merged_parts) > 100:
            raise PipelineError(f"Merge produced too many parts: {len(merged_parts)}")

        face_map = context.decomposition_dir / "final_face2cls.json"
        refined_again = part_refine_after_merge_main(
            str(context.mesh_path),
            str(face_map),
            save_dir=str(context.decomposition_dir),
            use_relative_threshold=(
                options.part_refine_after_merge_use_relative_threshold
            ),
        )
        final_parts = post_process_main(
            refined_again,
            str(context.mesh_path),
            save_dir=str(context.decomposition_dir),
            use_relative_threshold=options.post_process_use_relative_threshold,
        )
        record = (
            f"sam_3d_masks: {len(sam_masks)}\n"
            f"partfield_match: {best_cluster_file.name}\n"
            f"refined_parts: {len(new_parts)}\n"
            f"merged_parts: {len(merged_parts)}\n"
            f"refined_again_parts: {len(refined_again)}\n"
            f"final_parts: {len(final_parts)}\n"
            f"merge_strategy: {merge_flag}\n"
        )
        (context.decomposition_dir / "record.txt").write_text(record, encoding="utf-8")


class ExportPartsStage(Stage):
    number = 3
    name = "export_parts"
    description = "Export decomposed OBJ files and render per-part images."
    dependencies = ("decompose",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing = super().missing_inputs(context)
        result = context.decomposition_dir / "final_post_face2cls.json"
        if not _json_file_valid(result):
            missing.append(str(result))
        return missing

    def run(self, context: EntityContext) -> None:
        from utils.glb2part import glb_to_part_main
        from utils.render_parts_ori import render_parts_ori_main
        from utils.render_parts_single import render_parts_single_main

        result_file = context.decomposition_dir / "final_post_face2cls.json"
        face_to_part = json.loads(result_file.read_text(encoding="utf-8"))
        part_count = len(set(face_to_part.values()))
        if part_count < 1:
            raise PipelineError("The final face-to-part mapping contains no parts.")
        if part_count >= context.config.pipeline.max_parts:
            raise PipelineError(
                f"Part count {part_count} exceeds max_parts={context.config.pipeline.max_parts}."
            )

        allowed = (context.decomposition_dir,)
        for name in ("imgs", "objs", "glbs"):
            _safe_rmtree(context.decomposition_dir / name, allowed_roots=allowed)

        glb_to_part_main(
            str(context.mesh_path),
            str(result_file),
            str(context.decomposition_dir),
        )
        previous = os.environ.get("UNIPHYS_BLENDER")
        os.environ["UNIPHYS_BLENDER"] = context.config.runtime.blender
        try:
            render_parts_ori_main(str(context.decomposition_dir), view_index=0)
            render_parts_ori_main(str(context.decomposition_dir), view_index=1)
            render_parts_single_main(str(context.decomposition_dir))
        finally:
            if previous is None:
                os.environ.pop("UNIPHYS_BLENDER", None)
            else:
                os.environ["UNIPHYS_BLENDER"] = previous
        if context.config.pipeline.remove_part_glbs_after_render:
            _safe_rmtree(context.decomposition_dir / "glbs", allowed_roots=allowed)

    def outputs_valid(self, context: EntityContext) -> bool:
        obj_dir = context.decomposition_dir / "objs"
        image_dir = context.decomposition_dir / "imgs"
        if not obj_dir.is_dir() or not image_dir.is_dir():
            return False
        part_ids = sorted(path.stem for path in obj_dir.glob("*.obj"))
        if not part_ids:
            return False
        return all(
            (image_dir / f"{part_id}_ori_0.png").is_file()
            and (image_dir / f"{part_id}_ori_1.png").is_file()
            and (image_dir / f"{part_id}_ori_single.png").is_file()
            for part_id in part_ids
        )


class AnnotateBasicStage(Stage):
    number = 4
    name = "annotate_basic"
    description = "Annotate object and part physical properties with a vision model."
    dependencies = ("export_parts",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing = super().missing_inputs(context)
        if not ExportPartsStage().outputs_valid(context):
            missing.append("export_parts stage outputs")
        if not context.config.gpt.basic_prompt.is_file():
            missing.append(str(context.config.gpt.basic_prompt))
        if not os.environ.get(context.config.gpt.api_key_env):
            missing.append(f"environment variable {context.config.gpt.api_key_env}")
        return missing

    def run(self, context: EntityContext) -> None:
        from utils.physics_basic_annotate import gpt_phys_basic_annotation_main

        gpt_phys_basic_annotation_main(
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir),
            context.entity,
            context.config.gpt.as_legacy_dict(),
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        return _json_file_valid(
            context.gpt_output_dir / "gpt_basic_annotation" / f"{context.entity}.json"
        )


class BuildKinematicGraphStage(Stage):
    number = 5
    name = "build_kinematic_graph"
    description = "Build the object kinematic graph from basic annotations."
    dependencies = ("annotate_basic",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        path = (
            context.gpt_output_dir / "gpt_basic_annotation" / f"{context.entity}.json"
        )
        return [] if _json_file_valid(path) else [str(path)]

    def run(self, context: EntityContext) -> None:
        from utils.generate_kinematic_graph import generate_kg_main

        generate_kg_main(
            str(context.mesh_path), str(context.gpt_output_dir), context.entity
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        base = context.gpt_output_dir / "kg_res" / context.entity
        return all(
            _json_file_valid(base / name)
            for name in (
                "motion_graph.json",
                "label2name.json",
                "motion_node_dependency.json",
                "kinematic_info.json",
            )
        )


class ProposeKinematicsStage(Stage):
    number = 6
    name = "propose_kinematics"
    description = "Generate candidate joint axes and pivots."
    dependencies = ("build_kinematic_graph",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        path = (
            context.gpt_output_dir / "kg_res" / context.entity / "kinematic_info.json"
        )
        return [] if _json_file_valid(path) else [str(path)]

    def run(self, context: EntityContext) -> None:
        from utils.physics_kinematic_annotate import kinematic_annotation_main

        kinematic_annotation_main(
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir),
            context.entity,
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        prismatic = _expected_motion_parts(context, "B")
        revolute = _expected_motion_parts(context, "C")
        return all(
            (
                context.gpt_output_dir
                / "obj_prismatic"
                / context.entity
                / part
                / "axis.npy"
            ).is_file()
            for part in prismatic
        ) and all(
            (
                context.gpt_output_dir
                / "obj_revolute"
                / context.entity
                / part
                / "axis/axis.npy"
            ).is_file()
            and (
                context.gpt_output_dir
                / "obj_revolute"
                / context.entity
                / part
                / "pivot/pivot.npy"
            ).is_file()
            for part in revolute
        )


class RenderAxesStage(Stage):
    number = 7
    name = "render_axes"
    description = "Render candidate joint axes and pivot points."
    dependencies = ("propose_kinematics",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        return (
            []
            if ProposeKinematicsStage().outputs_valid(context)
            else ["kinematic candidates"]
        )

    def run(self, context: EntityContext) -> None:
        from utils.blender_axis_main import render_axis_main

        previous = os.environ.get("UNIPHYS_BLENDER")
        os.environ["UNIPHYS_BLENDER"] = context.config.runtime.blender
        try:
            render_axis_main(
                str(context.gpt_output_dir), "obj_revolute", context.entity
            )
            render_axis_main(
                str(context.gpt_output_dir), "obj_prismatic", context.entity
            )
        finally:
            if previous is None:
                os.environ.pop("UNIPHYS_BLENDER", None)
            else:
                os.environ["UNIPHYS_BLENDER"] = previous

    def outputs_valid(self, context: EntityContext) -> bool:
        for part in _expected_motion_parts(context, "B"):
            images = (
                context.gpt_output_dir
                / "obj_prismatic"
                / context.entity
                / part
                / "imgs"
            )
            if len(list(images.glob("*.png"))) < 4:
                return False
        for part in _expected_motion_parts(context, "C"):
            base = context.gpt_output_dir / "obj_revolute" / context.entity / part
            if len(list((base / "axis/imgs").glob("*.png"))) < 4:
                return False
            if len(list((base / "pivot/imgs").glob("*.png"))) < 2:
                return False
        return True


class AnnotateRevoluteStage(Stage):
    number = 8
    name = "annotate_revolute"
    description = "Select revolute axes and pivots with a vision model."
    dependencies = ("render_axes",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing: list[str] = []
        if not RenderAxesStage().outputs_valid(context):
            missing.append("rendered kinematic candidates")
        if not context.config.gpt.axis_prompt.is_file():
            missing.append(str(context.config.gpt.axis_prompt))
        if not os.environ.get(context.config.gpt.api_key_env):
            missing.append(f"environment variable {context.config.gpt.api_key_env}")
        return missing

    def run(self, context: EntityContext) -> None:
        from utils.physics_revolute_gpt import gpt_phys_revolute_annotation_main

        gpt_phys_revolute_annotation_main(
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir),
            context.entity,
            context.config.gpt.as_legacy_dict(),
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        base = context.gpt_output_dir / "gpt_revolute_annotation" / context.entity
        return all(
            _json_file_valid(base / f"{part}.json")
            for part in _expected_motion_parts(context, "C")
        )


class AnnotatePrismaticStage(Stage):
    number = 9
    name = "annotate_prismatic"
    description = "Select prismatic axes with a vision model."
    dependencies = ("render_axes",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        return AnnotateRevoluteStage().missing_inputs(context)

    def run(self, context: EntityContext) -> None:
        from utils.physics_prismatic_gpt import gpt_phys_prismatic_annotation_main

        gpt_phys_prismatic_annotation_main(
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir),
            context.entity,
            context.config.gpt.as_legacy_dict(),
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        base = context.gpt_output_dir / "gpt_prismatic_annotation" / context.entity
        return all(
            _json_file_valid(base / f"{part}.json")
            for part in _expected_motion_parts(context, "B")
        )


class GenerateMujocoStage(Stage):
    number = 10
    name = "generate_mujoco"
    description = "Generate MuJoCo XML models from joint annotations."
    dependencies = ("annotate_revolute", "annotate_prismatic")

    def missing_inputs(self, context: EntityContext) -> list[str]:
        missing: list[str] = []
        if not AnnotateRevoluteStage().outputs_valid(context):
            missing.append("revolute annotations")
        if not AnnotatePrismaticStage().outputs_valid(context):
            missing.append("prismatic annotations")
        return missing

    def run(self, context: EntityContext) -> None:
        from utils.generate_kg_mujoco_xml import generate_kg_mujoco_xml

        generate_kg_mujoco_xml(
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir),
            context.entity,
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        expected = len(_expected_motion_parts(context, "B")) + len(
            _expected_motion_parts(context, "C")
        )
        if expected == 0:
            return BuildKinematicGraphStage().outputs_valid(context)
        xml_dir = context.gpt_output_dir / "kg_xml" / context.entity
        return len(list(xml_dir.glob("*.xml"))) >= expected


class ValidateBasicStage(Stage):
    number = 11
    name = "validate_basic"
    description = "Validate predicted basic physical properties."
    dependencies = ("annotate_basic",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        annotation = (
            context.gpt_output_dir / "gpt_basic_annotation" / f"{context.entity}.json"
        )
        return [] if _json_file_valid(annotation) else [str(annotation)]

    def run(self, context: EntityContext) -> None:
        from utils.validate_basic_phys import basic_validation_process_main

        basic_validation_process_main(
            context.entity,
            str(context.mesh_path),
            str(context.config.dataset.decomposition_root),
            str(context.gpt_output_dir / "gpt_basic_annotation"),
            str(context.gpt_output_dir / "basic_check"),
        )

    def outputs_valid(self, context: EntityContext) -> bool:
        return _json_file_valid(
            context.gpt_output_dir / "basic_check" / f"{context.entity}.json"
        )


class ValidateSimulationStage(Stage):
    number = 12
    name = "validate_simulation"
    description = "Run MuJoCo simulation validation for generated XML models."
    dependencies = ("generate_mujoco",)

    def missing_inputs(self, context: EntityContext) -> list[str]:
        expected = len(_expected_motion_parts(context, "B")) + len(
            _expected_motion_parts(context, "C")
        )
        if expected == 0:
            return []
        xml_dir = context.gpt_output_dir / "kg_xml" / context.entity
        return [] if list(xml_dir.glob("*.xml")) else [f"MuJoCo XML files in {xml_dir}"]

    def run(self, context: EntityContext) -> None:
        from utils.validate_mujoco_simulation import mujoco_sim_validation_main

        if not list((context.gpt_output_dir / "kg_xml" / context.entity).glob("*.xml")):
            return
        mujoco_sim_validation_main(str(context.gpt_output_dir), context.entity)

    def outputs_valid(self, context: EntityContext) -> bool:
        expected = len(_expected_motion_parts(context, "B")) + len(
            _expected_motion_parts(context, "C")
        )
        if expected == 0:
            return BuildKinematicGraphStage().outputs_valid(context)
        xml_dir = context.gpt_output_dir / "kg_xml" / context.entity
        xml_files = list(xml_dir.glob("*.xml"))
        return bool(xml_files) and all(
            _json_file_valid(xml.with_suffix(".json")) for xml in xml_files
        )


def create_registry() -> StageRegistry:
    """Build the canonical pipeline registry."""

    return StageRegistry(
        (
            RenderStage(),
            DecomposeStage(),
            ExportPartsStage(),
            AnnotateBasicStage(),
            BuildKinematicGraphStage(),
            ProposeKinematicsStage(),
            RenderAxesStage(),
            AnnotateRevoluteStage(),
            AnnotatePrismaticStage(),
            GenerateMujocoStage(),
            ValidateBasicStage(),
            ValidateSimulationStage(),
        )
    )
