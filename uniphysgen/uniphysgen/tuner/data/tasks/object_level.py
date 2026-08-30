from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseTaskHandler


class ObjectLevelTask(BaseTaskHandler):
    """Object-level regression via JSON generation.

    Primary targets: volume and mass.
    """

    TASK_NAME = "object_level"

    DEFAULT_PROMPT = (
        "[OBJECT_LEVEL]\n"
        "Estimate object identity and object-level physical properties.\n\n"

        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n"
        "<image>\n"

        "# Notes\n"
        "- volume is a 3-number array [L, W, H] in centimeters (cm).\n"
        "- mass is in kilograms (kg).\n"

        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"object_name\": string|null,\n"
        "  \"category\": string|null,\n"
        "  \"volume\": [number, number, number]|null,\n"
        "  \"mass\": number|null\n"
        "}\n"
    )

    DEFAULT_PROMPT2 = (
        "[OBJECT_LEVEL]\n"
        "Estimate object identity and object-level physical properties.\n\n"

        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<image>\n"

        "# Notes\n"
        "- volume is a 3-number array [L, W, H] in centimeters (cm).\n"
        "- mass is in kilograms (kg).\n"

        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"object_name\": string|null,\n"
        "  \"category\": string|null,\n"
        "  \"volume\": [number, number, number]|null,\n"
        "  \"mass\": number|null\n"
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
        object = feature.get("object_info") or {}
        img_pth = object.get("object_img") or ""
        if img_pth == "" or img_pth is None:
            return mm_feature
        images = [img_pth]
        images = [p for p in images if p]
        images = cls.find_medias(images, load_from=load_from, media_dir=media_dir) or []
        images = cls.validate_existing_paths(images, kind="image") or []
        mm_feature["_images"] = images
        # if len(images) == 0:
        #     print(images, "none", object["object_ply"])
        return mm_feature

    @classmethod
    def build_prompt(cls) -> str:
        return cls.DEFAULT_PROMPT2

    @classmethod
    def format_label(cls, feature: Dict[str, Any]) -> str:
        obj = feature.get("object_info") or {}

        # If you have explicit volume in raw, use it. Otherwise you can compute
        # from dimensions in your template/converter and store it in feature.
        object_name = obj.get("object_name")
        category = obj.get("category")
        volume = obj.get("dimension")
        mass = obj.get("object_mass")

        out = {
            "object_name": object_name,
            "category": category,
            "volume": volume,
            "mass": mass,
        }
        return json.dumps(out, ensure_ascii=False)
