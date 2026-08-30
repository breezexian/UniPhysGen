from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseTaskHandler


class PhysicsTask(BaseTaskHandler):
    """Physics (part-level) task.

    Prompt is a skeleton with placeholders. Converter/template can inject
    structured fields and multimodal placeholders.
    """

    TASK_NAME = "physics"

    @staticmethod
    def maybe_mask_basic_description(text: str | None, *, p: float) -> str | None:
        """Optionally mask the *input-side* basic description.

        Keep labels deterministic. Apply stochastic masking when constructing prompt inputs.
        """
        if text is None or p <= 0:
            return text
        # Local import to avoid introducing global RNG side effects at import time.
        import random

        return None if random.random() < p else text

    """
        "[PHYSICS]\n"
        "Estimate the part identity and its semantic descriptions\n"
        "by jointly reasoning over the part geometry and the full object point cloud as context\n"
        "to disambiguate the part’s role and motion,\n"
        "then infer plausible physical properties consistent with the geometry and semantics.\n\n"
    """
    DEFAULT_PROMPT = (
        "[PHYSICS]\n"
        "Estimate part identity, semantic descriptions, and physical properties.\n\n"

        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n"
        "<image>\n"

        "# Notes\n"
        "- affordance: smaller value means higher affordance.\n"
        "- motion_type:\n"
        "  A = contact-only (no relative motion),\n"
        "  B = translation,\n"
        "  C = rotation,\n"
        "  D = rigid (fixed, no motion).\n"
        "- Units: density in g/cm^3 (if applicable), young (Young's modulus) in GPa, hardness in HV, poisson (Poisson's ratio) unitless, friction unitless.\n"

        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"part_identity\": {\n"
        "    \"part_name\": string|null,\n"
        "    \"motion_type\": \"A\"|\"B\"|\"C\"|\"D\"|null,\n"
        "  },\n"
        "  \"semantic_description\": {\n"
        "    \"basic\": string|null,\n"
        "    \"functional\": string|null,\n"
        "    \"movement\": string|null,\n"
        "    \"grasp\": string|null\n"
        "  },\n"
        "  \"physical_properties\": {\n"
        "    \"material\": string|null,\n"
        "    \"density\": number|null,\n"
        "    \"young\": number|null,\n"
        "    \"hardness\": number|null,\n"
        "    \"poisson\": number|null,\n"
        "    \"friction\": number|null,\n"
        "    \"graspable\": boolean|null,\n"
        "    \"affordance\": 1|2|3|4|5|6|7|8|9|10|null\n"
        "  }\n"
        "}\n"
    )

    SYSTEM_PROMPT = "You are a helpful assistant."

    @classmethod
    def build_system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT

    @classmethod
    def build_mm_features(
            cls,
            feature: Dict[str, Any],
            *,
            media_dir: str | None = None,
            load_from: str = "file",
    ) -> Dict[str, Any]:
        mm_feature = super().build_mm_features(feature, media_dir=media_dir, load_from=load_from)
        return mm_feature

    @classmethod
    def build_prompt(cls) -> str:
        return cls.DEFAULT_PROMPT

    @classmethod
    def format_label(cls, feature: Dict[str, Any]) -> str:
        """Format part-level labels as strict JSON.

        Expected raw fields (per dataset examples):
        - feature["part_info"]: {
            part_name, affordance, graspable,
            basic_description, functional_description, movement_description, grasp_description
          }
        - feature["basic_info"]: {material, density, young, hardness, poisson, friction}
        """

        part_info = feature.get("part_info") or {}
        basic_info = feature.get("basic_info") or {}
        kin_info = feature.get("kinematic_info") or {}

        def _round2(x: Any) -> Any:
            if x is None:
                return None
            # keep bool as-is
            if isinstance(x, bool):
                return x
            # keep integers as-is
            if isinstance(x, int):
                return x
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return x
            return round(xf, 2)

        # motion_types can be missing/None/empty; fall back to single key if present.
        motion_types = kin_info.get("motion_types")
        motion_type = "D"
        if isinstance(motion_types, (list, tuple)) and motion_types:
            motion_type = motion_types[0]
        if isinstance(motion_type, str):
            motion_type = motion_type.strip()
        if motion_type not in {"A", "B", "C", "D"}:
            motion_type = "D"

        out = {
            "part_identity": {
                "part_name": part_info.get("part_name"),
                "motion_type": motion_type,
            },
            "semantic_description": {
                "basic": part_info.get("basic_description"),
                "functional": part_info.get("functional_description"),
                "movement": part_info.get("movement_description"),
                "grasp": part_info.get("grasp_description"),
            },
            "physical_properties": {
                "material": basic_info.get("material"),
                "density": _round2(basic_info.get("density")),
                "young": _round2(basic_info.get("young")),
                # hardness is typically an integer HV; keep as number but do not force int.
                "hardness": _round2(basic_info.get("hardness")),
                "poisson": _round2(basic_info.get("poisson")),
                "friction": _round2(basic_info.get("friction")),
                "graspable": part_info.get("graspable"),
                "affordance": part_info.get("affordance"),
            },
        }
        return json.dumps(out, ensure_ascii=False)
