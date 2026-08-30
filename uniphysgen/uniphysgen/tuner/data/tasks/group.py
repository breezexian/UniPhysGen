from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseTaskHandler


class GroupTask(BaseTaskHandler):
    """Part grouping task.

    This task typically requires candidate parts (or a part list) for the same object.
    Keep placeholders so you can later inject candidates in your template.
    """

    TASK_NAME = "group"

    DEFAULT_PROMPT = (
        "[GROUP]\n"
        "Predict the motion-coupled part group(s): which parts will move together when the target part moves.\n\n"

        "# Inputs (placeholders)\n"
        "- parts(id,position): {{PART_LIST}}\n\n"

        "# Multimodal Inputs\n"
        "<object_point_cloud>\n"
        "<part_point_cloud>\n"
        "<image>\n\n"

        "# Constraints\n"
        "- \"members\" must be selected ONLY from the part ids provided in the input parts list.\n"
        "- Do NOT generate new ids.\n"
        "- Each id should appear at most once.\n"
        "- The output must be a subset of the input part ids.\n"
        "- \"members\" must include the target part itself.\n\n"

        "# Output (JSON only)\n"
        "Return a single JSON object with the following schema (no extra keys):\n"
        "{\n"
        "  \"members\": [number]\n"
        "}\n\n"
    )

    SYSTEM_PROMPT = (
        "You are a multimodal assistant. Follow the instructions strictly.\n"
        "- Output JSON only. Do not wrap in markdown. Do not add extra keys.\n"
        "- If an input modality token is absent, do not assume it exists.\n"
        "- Modality tokens (if present) are placeholders for model inputs: "
        "<object_point_cloud>, <image_3panel>.\n"
        "- members must be a JSON array of part identifiers (strings).\n"
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
        """Format motion-coupled group labels as strict JSON.

        Uses:
          - feature["kinematic_info"]["motion_info"]["dependency"]: list[int]
          - feature["part_info"]["part_label_to_name"]: dict[str, str]

                Output is a single motion group as a list of part ids.
                If an id in dependency is missing from part_label_to_name, it is dropped.
        """

        part = feature.get("part_info") or {}
        kin = feature.get("kinematic_info") or {}
        motion_info = kin.get("motion_info") or {}

        part_label_to_name = part.get("part_label_to_name") or {}
        if not isinstance(part_label_to_name, dict):
            part_label_to_name = {}

        dep = motion_info.get("dependency") or []
        if not isinstance(dep, (list, tuple)):
            dep = []

        def _id_to_member(x: Any) -> int | None:
            try:
                pid = int(x)
            except (TypeError, ValueError):
                return None
            name = part_label_to_name.get(str(pid))
            if not name:
                return None
            return pid

        members: list[int] = []
        seen: set[int] = set()
        for x in dep:
            m = _id_to_member(x)
            if m is None:
                continue
            if m in seen:
                continue
            members.append(m)
            seen.add(m)

        out = {"members": members}
        return json.dumps(out, ensure_ascii=False)
