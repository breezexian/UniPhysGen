from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import BaseTaskHandler


class MotionTask(BaseTaskHandler):
    """Kinematics/motion task.

    Even if you later supervise motion with a dedicated head, keeping a JSON
    label formatter is useful for text-only baselines and debugging.
    """

    TASK_NAME = "motion"

    # Legacy prompt kept for backward compatibility.
    DEFAULT_PROMPT_REG = (
        "[MOTION]\n"
        "Predict the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n"
    )

    # New prompt in a stricter, schema-first format.
    DEFAULT_PROMPT2 = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized and shifted to [0, 2].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is a 3-number unit vector [x, y, z] when provided.\n"
        "- pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "  - If motion_type is B, pivot must be [0, 0, 0].\n"
        "  - If motion_type is C, pivot is the rotation center.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": [number, number, number]|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )

    # New prompt in a stricter, schema-first format.
    DEFAULT_PROMPT = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized and shifted to [0, 2].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is a 3-number unit vector [x, y, z] when provided.\n"
        "- pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": [number, number, number]|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )

    # New prompt in a stricter, schema-first format.
    DEFAULT_PROMPT_NO_PIVOT = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized and shifted to [0, 2].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is a 3-number unit vector [x, y, z] when provided.\n"
        "- For B (translation), pivot must be null because translation does not have a fixed pivot.\n"
        "- For C (rotation), pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": [number, number, number]|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )

    DEFAULT_PROMPT_ORI = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized to [-1, 1].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is a 3-number unit vector [x, y, z] when provided.\n"
        "- pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": [number, number, number]|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )
    

    # Spherical-axis prompt for newer datasets/plugins.
    # axis is represented as integer degrees in spherical coordinates.
    DEFAULT_PROMPT3 = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized and shifted to [0, 2].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is represented by spherical coordinates (degrees):\n"
        "  - theta: polar angle from +Z, integer in [0, 180].\n"
        "  - phi: azimuth angle around Z, integer in [0, 360).\n"
        "- pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": {\"theta\": integer, \"phi\": integer}|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )

    DEFAULT_PROMPT3_ORI = (
        "[MOTION]\n"
        "Estimate the kinematic motion of the target part.\n\n"
        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n\n"
        "# Notes\n"
        "- The input point clouds are AABB-normalized to [-1, 1].\n"
        "- motion_type uses letters: B = translation (prismatic), C = rotation (revolute).\n"
        "- axis is represented by spherical coordinates (degrees):\n"
        "  - theta: polar angle from +Z, integer in [0, 180].\n"
        "  - phi: azimuth angle around Z, integer in [0, 360).\n"
        "- pivot is a 3-number point [x, y, z] in the same coordinate frame as the point cloud.\n"
        "- range is a 2-number array [min, max].\n"
        "- For C (revolute), range is normalized by 2π (dimensionless, i.e., angle/(2π)).\n"
        "- For B (prismatic), range is in normalized length units in the AABB-normalized coordinate frame.\n\n"
        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"motion_type\": \"B\"|\"C\"|null,\n"
        "  \"axis\": {\"theta\": integer, \"phi\": integer}|null,\n"
        "  \"pivot\": [number, number, number]|null,\n"
        "  \"range\": [number, number]|null\n"
        "}\n"
    )

    SYSTEM_PROMPT = (
        "You are a multimodal assistant. Follow the instructions strictly.\n"
        "- Output JSON only. Do not wrap in markdown. Do not add extra keys.\n"
        "- If an input modality token is absent, do not assume it exists.\n"
        "- Modality tokens (if present) are placeholders for model inputs: "
        "<object_point_cloud>, <part_point_cloud>, <image_3panel>.\n"
        "- The input point clouds are AABB-normalized and shifted to [0, 2].\n"
        "- axis must be a unit vector when provided.\n"
        "- pivot is a 3D point in the same coordinate frame as the point cloud (if used).\n"
        "- For revolute joints, range is in radians.\n"
        "- For prismatic joints, range is in normalized length units.\n"
    )

    # Keep user override (may intentionally replace the detailed system prompt above).
    SYSTEM_PROMPT = "You are a helpful assistant."

    @classmethod
    def build_system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT

    @classmethod
    def build_mm_features(
            cls,
            feature: Dict[str, Any],
            *,
            media_dir: Optional[str] = None,
            load_from: str = "file",
    ) -> Dict[str, Any]:
        """Build multimodal feature fields for this task.

        Returns a dict that can be attached to the converted example, e.g.:
        {
          "images": [..],
          "part_point_clouds": [..],
          "object_point_clouds": [..],
          "motions": {...}
        }
        """
        mm_feature = super().build_mm_features(feature, media_dir=media_dir, load_from=load_from)
        return mm_feature

    @classmethod
    def build_prompt(cls) -> str:
        return cls.DEFAULT_PROMPT

    @classmethod
    def format_label(cls, feature: Dict[str, Any]) -> str:
        kin = feature.get("kinematic_info") or {}
        motion_info = kin.get("motion_info") or {}

        if not isinstance(motion_info, dict):
            raise ValueError(f"kinematic_info.motion_info must be a dict, got {type(motion_info)}")

        # Pick a valid motion key (same convention as mm_plugin):
        # - ignore 'dependency'
        # - value must be a dict
        # - axis must exist (not None)
        # - key must be B or C
        keys = [
            k
            for k in motion_info.keys()
            if k is not None
            and str(k).strip() != "dependency"
            and str(k).strip() in {"B", "C"}
            and isinstance(motion_info.get(k), dict)
            and motion_info.get(k).get("axis") is not None
        ]
        if len(keys) == 0:
            raise ValueError(
                f"Motion label missing valid motion_type key ('B' or 'C'). Got keys={list(motion_info.keys())}"
            )

        mtype_key = str(keys[0]).strip()
        m = motion_info.get(mtype_key) or {}

        # Map dataset-specific keys to a generic schema.
        # NOTE: your raw uses keys like "B"; joint type may be inferred from folder name
        # or stored elsewhere. Here we keep it as "other" unless you provide it.
        out = {
            "motion_type": mtype_key,
            "axis": m.get("axis"),
            "pivot": m.get("pos"),
            "range": m.get("range"),
        }
        return json.dumps(out, ensure_ascii=False)
