from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ...framework import logging

logger = logging.get_logger(__name__)

MediaType = Union[str, List[str]]


class BaseTaskHandler(ABC):
    """Base class for task handlers.

    Provides common media utilities so all tasks can share:
      - find_medias
      - validate_existing_paths
    """

    TASK_NAME: str

    @classmethod
    def build_system_prompt(cls) -> str:
        return ""

    @classmethod
    @abstractmethod
    def build_prompt(cls) -> str:
        raise NotImplementedError

    @classmethod
    def build_mm_features(
            cls,
            feature: Dict[str, Any],
            *,
            media_dir: Optional[str] = None,
            load_from: str = "file",
    ) -> Dict[str, Any]:
        """Build multimodal fields for the model input.

        Default implementation assumes the following schema:
          - images: [part_info.global_img_pth]
          - part/object point clouds: [part_info.obj_pth]
          - motions: kinematic_info.motion_info

        Tasks can override this when their raw schema differs.
        """
        object = feature.get("object_info") or {}
        part = feature.get("part_info") or {}
        kin = feature.get("kinematic_info") or {}

        img_pth = part.get("part_img")
        if img_pth is None:
            img_pth = ""
        img_name = os.path.basename(img_pth)
        label = img_name.split("_")[0]
        new_img_name = f"{label}_ori_0.png"
        img_pth = img_pth.replace(img_name, new_img_name)

        images = [img_pth]
        images = [p for p in images if p]

        part_pcds = [part.get("part_ply")]
        part_pcds = [p for p in part_pcds if p]

        obj_pcds = [object.get("object_ply")]
        obj_pcds = [p for p in obj_pcds if p]

        images = cls.find_medias(images, load_from=load_from, media_dir=media_dir) or []
        part_pcds = cls.find_medias(part_pcds, load_from=load_from, media_dir=media_dir) or []
        obj_pcds = cls.find_medias(obj_pcds, load_from=load_from, media_dir=media_dir) or []

        images = cls.validate_existing_paths(images, kind="image") or []
        part_pcds = cls.validate_existing_paths(part_pcds, kind="part_point_cloud") or []
        obj_pcds = cls.validate_existing_paths(obj_pcds, kind="object_point_cloud") or []

        return {
            "_images": images,
            "_part_point_clouds": part_pcds,
            "_object_point_clouds": obj_pcds,
            "_motions": {}
            # Filter out None values added by load_dataset schema inference
            # "_motions": {k: v for k, v in (kin.get("motion_info") or {}).items() if v is not None},
        }

    @classmethod
    @abstractmethod
    def format_label(cls, feature: Dict[str, Any]) -> str:
        raise NotImplementedError

    @staticmethod
    def find_medias(
            medias: Union[MediaType, List[MediaType], None],
            *,
            load_from: str = "file",
            media_dir: Optional[str] = None,
    ) -> Optional[List[MediaType]]:
        return find_medias(medias, load_from=load_from, media_dir=media_dir)

    @staticmethod
    def validate_existing_paths(
            paths: Optional[List[str]],
            *,
            keep_missing: bool = True,
            kind: str = "media",
    ) -> Optional[List[str]]:
        return validate_existing_paths(paths, keep_missing=keep_missing, kind=kind)


def find_medias(
        medias: Union[MediaType, List[MediaType], None],
        *,
        load_from: str = "file",
        media_dir: Optional[str] = None,
) -> Optional[List[MediaType]]:
    """Normalize and validate media paths.

    - Accepts a single media, a list of medias, or None.
    - Supports nested list[str] (e.g., video frames).
    - If load_from == "file" and media_dir is provided, will try to join
      relative paths with media_dir and replace them if the file exists.
    - If a joined path does not exist, keeps the original and logs a warning.

    Returns:
        A normalized list of medias, or None.
    """

    if medias is None:
        return None
    if not isinstance(medias, list):
        medias_list: List[MediaType] = [medias]
    else:
        if len(medias) == 0:
            return None
        medias_list = medias[:]

    if load_from in ["file"] and media_dir:
        first = medias_list[0]
        if isinstance(first, str):
            for i in range(len(medias_list)):
                m = medias_list[i]
                if not isinstance(m, str):
                    continue
                media_path = os.path.join(media_dir, m)
                if os.path.isfile(media_path):
                    medias_list[i] = media_path
                else:
                    logger.warning_rank0_once(
                        f"Media {m} does not exist in `media_dir`. Use original path."
                    )
        elif isinstance(first, list):
            for i in range(len(medias_list)):
                frames = medias_list[i]
                if not isinstance(frames, list):
                    continue
                for j in range(len(frames)):
                    f = frames[j]
                    if not isinstance(f, str):
                        continue
                    media_path = os.path.join(media_dir, f)
                    if os.path.isfile(media_path):
                        frames[j] = media_path
                    else:
                        logger.warning_rank0_once(
                            f"Media {f} does not exist in `media_dir`. Use original path."
                        )

    return medias_list


def validate_existing_paths(
        paths: Optional[List[str]],
        *,
        keep_missing: bool = True,
        kind: str = "media",
) -> Optional[List[str]]:
    """Optionally filter/validate a list of file paths."""

    if not paths:
        return None

    out: List[str] = []
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            out.append(p)
        else:
            logger.warning_rank0_once(f"{kind} path does not exist: {p}")
            if keep_missing:
                out.append(p)

    return out or None


def get_by_path(obj: Any, path: str, default: Any = None) -> Any:
    """Get nested value by dot-path.

    Supports dict keys and list indices (e.g., "a.b.0.c").
    """

    if not path:
        return default
    cur: Any = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
            else:
                return default
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return default
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return default
        else:
            return default
    return cur
