"""Lazy compatibility exports for legacy UniPhys utility functions.

Importing ``utils`` used to import every optional dependency eagerly. The
public names remain available, but modules are now loaded only when a caller
requests a specific function. New code should prefer direct submodule imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "read_mesh": (".mesh_handle", "read_mesh"),
    "face_areas": (".mesh_handle", "face_areas"),
    "get_face_areas_list": (".mesh_handle", "get_face_areas_list"),
    "build_mesh_from_part": (".mesh_handle", "build_mesh_from_part"),
    "build_mesh_from_face2cls": (".mesh_handle", "build_mesh_from_face2cls"),
    "sam_3d_mask_main": (".sam_3d_mask", "sam_3d_mask_main"),
    "hm_main": (".hungarian_match", "hm_main"),
    "part_refine_main": (".part_refine", "part_refine_main"),
    "merge_parts_main": (".merge_part", "merge_parts_main"),
    "post_process_main": (".post_process", "post_process_main"),
    "render_parts_ori_main": (".render_parts_ori", "render_parts_ori_main"),
    "render_parts_color_main": (".render_parts_color", "render_parts_color_main"),
    "glb_to_part_main": (".glb2part", "glb_to_part_main"),
    "render_parts_single_main": (".render_parts_single", "render_parts_single_main"),
    "part_refine_after_merge_main": (
        ".part_refine_after_merge",
        "part_refine_after_merge_main",
    ),
    "gpt_phys_basic_annotation_main": (
        ".physics_basic_annotate",
        "gpt_phys_basic_annotation_main",
    ),
    "kinematic_annotation_main": (
        ".physics_kinematic_annotate",
        "kinematic_annotation_main",
    ),
    "gpt_phys_revolute_annotation_main": (
        ".physics_revolute_gpt",
        "gpt_phys_revolute_annotation_main",
    ),
    "gpt_phys_prismatic_annotation_main": (
        ".physics_prismatic_gpt",
        "gpt_phys_prismatic_annotation_main",
    ),
    "render_axis_main": (".blender_axis_main", "render_axis_main"),
    "apriori_merge_parts_main": (".apriori_merge_parts", "apriori_merge_parts_main"),
    "generate_kg_main": (".generate_kinematic_graph", "generate_kg_main"),
    "generate_kg_mujoco_xml": (".generate_kg_mujoco_xml", "generate_kg_mujoco_xml"),
    "exr_combine_main": (".exr_combine", "exr_combine_main"),
    "read_exr": (".exr_combine", "read_exr"),
    "mujoco_sim_validation_main": (
        ".validate_mujoco_simulation",
        "mujoco_sim_validation_main",
    ),
    "basic_validation_process_main": (
        ".validate_basic_phys",
        "basic_validation_process_main",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
