import os
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Union, Sequence, Optional, Tuple

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from uniphysgen.pcd import load_o3d_pcd, get_points_and_colors
from uniphysgen.pcd.transform import Compose
from uniphysgen.tuner.data.tasks.motion import MotionTask

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer

    PointCloudInput = Union[str, dict, NDArray]

POINT_S_TOKEN = os.environ.get("POINT_S_TOKEN", "<|point_start|>")
POINT_E_TOKEN = os.environ.get("POINT_E_TOKEN", "<|point_end|>")
POINT_CLOUD_PLACEHOLDER = os.environ.get("POINT_CLOUD_PLACEHOLDER", "<point_cloud>")

# PhysLLM placeholders (dual point clouds + optional image)
PART_POINT_CLOUD_PLACEHOLDER = os.environ.get(
    "PART_POINT_CLOUD_PLACEHOLDER", "<part_point_cloud>"
)
OBJECT_POINT_CLOUD_PLACEHOLDER = os.environ.get(
    "OBJECT_POINT_CLOUD_PLACEHOLDER", "<object_point_cloud>"
)
PART_POINT_S_TOKEN = os.environ.get("PART_POINT_S_TOKEN", "<|part_point_start|>")
PART_POINT_E_TOKEN = os.environ.get("PART_POINT_E_TOKEN", "<|part_point_end|>")
OBJECT_POINT_S_TOKEN = os.environ.get(
    "OBJECT_POINT_S_TOKEN", "<|object_point_start|>"
)
OBJECT_POINT_E_TOKEN = os.environ.get(
    "OBJECT_POINT_E_TOKEN", "<|object_point_end|>"
)

IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")
IMAGE_S_TOKEN = os.environ.get("IMAGE_S_TOKEN", "<|vision_start|>")
IMAGE_E_TOKEN = os.environ.get("IMAGE_E_TOKEN", "<|vision_end|>")
NORMALIZATION_PRESET = {
    "world": (0.0, 2.0),
}


class UniPhysGenPlugin:
    """Multimodal plugin for PhysLLM.

    Supports:
      - dual point clouds: part-level and object-level
      - optional images (paths only in this MVP)
      - motion labels synchronized with the same geometric transform

    Notes:
      - part/object point clouds MUST share the same sampled transform per sample.
      - motion labels MUST be transformed with the same transform.
    """

    def __init__(
            self,
            point_token: str = "<|point_pad|>",
            image_token: str = "<|image_pad|>",
            num_bins: int = 400,
            do_augmentation: bool = False,
            random_rotation: bool = False,
            random_xyz_rotation: bool = False,
            random_scaling: bool = False,
            use_spherical_axis: bool = False,
            share_grid_origin: bool = True,
            grid_sample_mode: str = "train",
            image_size: int = 224,
            image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.point_token = point_token
        self.image_token = image_token
        global_extent = NORMALIZATION_PRESET["world"]
        self.num_bins = num_bins
        self.grid_size = (global_extent[1] - global_extent[0]) / self.num_bins
        self.do_augmentation = do_augmentation
        self.random_rotation = random_rotation
        # If True, when random_rotation is enabled, randomly choose one axis from {x,y,z}
        # to apply a random rotation. If False, keep legacy behavior (z-axis only).
        self.random_xyz_rotation = random_xyz_rotation
        self.random_scaling = random_scaling
        self.use_spherical_axis = bool(use_spherical_axis)
        self.share_grid_origin = bool(share_grid_origin)
        self.augmentation = Compose(
            [
                dict(type="RandomColorGrayScale", p=0.05),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.75, ratio=0.1),
                dict(type="ChromaticJitter", p=0.8, std=0.05),
                dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                dict(type="RandomColorDrop", p=0.1, color_augment=0.0),
                dict(type="RandomJitter", sigma=0.025, clip=0.05, ratio=0.8, p=0.9),
                dict(type="RandomJitter", sigma=0.2, clip=0.2, ratio=0.05, p=0.85),
                dict(type="RandomJitter", sigma=0.4, clip=1.0, ratio=0.001, p=0.75),
                dict(type="RandomJitter", sigma=0.5, clip=4.0, ratio=0.0005, p=0.7),
                dict(
                    type="ElasticDistortion",
                    distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    p=[0.85, 0.5],
                ),
            ]
        )

        # Use "test" mode for GridSample when augmentation is off (inference),
        # so voxel sampling is deterministic (picks center point instead of random).
        self.transform = Compose(
            [
                # IMPORTANT: do NOT CenterShift per-cloud here.
                # We must keep part/object in the same coordinate frame.
                # CenterShift is applied once with a shared shift computed from the
                # object point cloud in get_mm_inputs().
                dict(type="NormalizeColor"),
                dict(
                    type="GridSample",
                    grid_size=self.grid_size,
                    hash_type="fnv",
                    mode=grid_sample_mode,
                    keys=("coord", "color", "normal"),
                    return_grid_coord=True,
                    max_grid_coord=self.num_bins,
                ),
            ]
        )
        self.image_size = int(image_size)
        self.image_mean = tuple(float(x) for x in image_mean)
        self.image_std = tuple(float(x) for x in image_std)

    @staticmethod
    def _vector_to_theta_phi_deg(vec: np.ndarray) -> Tuple[float, float]:
        """Convert a 3D direction vector to spherical angles (theta, phi) in degrees.

        Conventions:
          - theta: polar angle from +Z, in [0, 180]
          - phi: azimuth angle in XY-plane from +X toward +Y, in [0, 360)

        Canonicalization:
          - enforce z >= 0 by flipping the direction if z < 0 (axis sign ambiguity)
        """
        v = np.asarray(vec, dtype=np.float32).reshape(3)

        # axis direction is sign-ambiguous; canonicalize to z >= 0 for stability.
        if float(v[2]) < 0.0:
            v = -v

        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-12:
            raise ValueError("Zero-length or invalid vector")
        x, y, z = (v / n).tolist()

        theta = math.degrees(math.acos(float(z)))
        phi = math.degrees(math.atan2(float(y), float(x)))
        if phi < 0:
            phi += 360.0
        return float(theta), float(phi)

    @staticmethod
    def _should_flip_axis_canonical(vec: np.ndarray) -> bool:
        """Return True if we would flip the axis under the z>=0 canonicalization."""
        v = np.asarray(vec, dtype=np.float32).reshape(3)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-12:
            return False
        z = float((v / n)[2])
        return z < 0.0

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        """Load an image from path and return normalized float tensor (3, H, W)."""
        if Image is None:
            raise ImportError("PIL is required to load images. Please install pillow.")
        if not os.path.exists(image_path):
            return torch.zeros(3, self.image_size, self.image_size)
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            return torch.zeros(3, self.image_size, self.image_size)
        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
        mean = np.asarray(self.image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.image_std, dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / (std + 1e-12)
        return torch.as_tensor(arr)

    def _preprocess_point_cloud(self, point_cloud: dict, shared_grid_min_coord=None) -> np.ndarray:
        """Pre-process a single point cloud dict using the same transform pipeline.

        Args:
            point_cloud: dict with keys "coord", "color", "normal", etc.
            shared_grid_min_coord: if provided, pass to GridSample so that grid_coord
                is computed relative to a shared origin (e.g. from the object cloud).
                This keeps part and object Fourier positional encodings in the same
                reference frame.
        """
        if shared_grid_min_coord is not None:
            point_cloud["shared_grid_min_coord"] = shared_grid_min_coord
        point_cloud = self.transform(point_cloud)
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        color = point_cloud["color"]
        normal = point_cloud["normal"]
        assert len(coord) == len(xyz) == len(color)
        return np.concatenate([coord, xyz, color, normal], axis=1)

    @staticmethod
    def _compute_center_shift_from_object(points: np.ndarray, apply_z: bool = True) -> np.ndarray:
        """Compute the same shift as pcd.transform.CenterShift, but from object points only.

        We apply this *shared* shift to both object+part so their relative alignment is kept.
        """
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, _ = points.max(axis=0)
        if apply_z:
            shift = np.asarray([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=np.float32)
        else:
            shift = np.asarray([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0], dtype=np.float32)
        return shift

    def _compute_shared_grid_min_coord(self, point_cloud: dict) -> np.ndarray:
        """Compute the grid min_coord for a point cloud WITHOUT modifying it.

        This is used to derive a shared grid origin from the object point cloud,
        which is then passed to GridSample for both object and part point clouds,
        ensuring their grid_coords (and thus Fourier positional encodings) stay
        in the same reference frame.
        """
        scaled_coord = point_cloud["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        return grid_coord.min(0)

    def _regularize_point_clouds(
            self, point_clouds: Sequence["PointCloudInput"],
            shared_grid_min_coords: Optional[List[np.ndarray]] = None,
            **kwargs
    ) -> torch.Tensor:
        """Pad variable-length point clouds to a batch tensor (B, max_len, 9).

        Args:
            point_clouds: list of point cloud dicts.
            shared_grid_min_coords: if provided, a list of min_coord arrays (one per
                point cloud) to pass to GridSample for consistent grid_coord origins.
        """
        points_list: List[np.ndarray] = []
        max_len = 0
        for idx, point_cloud in enumerate(point_clouds):
            if not isinstance(point_cloud, dict):
                raise ValueError(
                    "Point cloud input must be a dictionary with 'name' and 'coord' keys."
                )
            smc = shared_grid_min_coords[idx] if shared_grid_min_coords is not None else None
            point_feats = self._preprocess_point_cloud(point_cloud, shared_grid_min_coord=smc, **kwargs)
            max_len = max(max_len, len(point_feats))
            points_list.append(point_feats)

        for i in range(len(points_list)):
            points_list[i] = np.pad(
                points_list[i],
                ((0, max_len - len(points_list[i])), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

        return torch.as_tensor(np.stack(points_list, axis=0))

    def _regularize_images(self, images: Sequence[Sequence[str]], use_image: bool) -> Optional[torch.Tensor]:
        """Convert list-of-list image paths to a batched tensor.

        MVP: expects 0 or 1 image per sample.
        Returns:
            images: (B, 3, H, W) or None
        """
        if len(images) == 0 or not use_image:
            return None
        # If any sample has no image, skip image modality for the whole batch.
        # This matches: no image -> use None and skip image token.
        # if any(len(x) == 0 for x in images):
        #     return None

        tensors: List[torch.Tensor] = []
        for paths in images:
            if len(paths) != 1:
                tensors.append(torch.zeros(3, self.image_size, self.image_size))
                # raise ValueError(
                #     f"PhysLLMPlugin MVP expects exactly one image per sample, got {len(paths)}"
                # )
                print(f"PhysLLMPlugin MVP expects exactly one image per sample, got {len(paths)}")
                continue
            tensors.append(self._load_image_tensor(paths[0]))
        return torch.stack(tensors, dim=0)

    @staticmethod
    def _as_list(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _sample_transform(self, points: np.ndarray) -> dict:
        """Sample a transform based on object points (shared for part/object)."""
        axis = "z"
        angle = 0.0
        if self.random_rotation:
            angle = float(np.random.random() * 2 * np.pi)
            if self.random_xyz_rotation:
                axis = str(np.random.choice(["x", "y", "z"]))
        else:
            # Legacy default: no rotation.
            angle = 0.0
        # NOTE: inputs are already AABB-normalized in get_mm_inputs().
        # Additional random scaling is optional; default off for sanity/consistency.
        scaling = np.random.uniform(0.75, 1.25) if self.random_scaling else 1.0
        # print(f"======lilililil axis {self.share_grid_origin, axis, angle}")
        if axis == "x":
            rotvec = np.array([angle, 0.0, 0.0], dtype=np.float32)
        elif axis == "y":
            rotvec = np.array([0.0, angle, 0.0], dtype=np.float32)
        else:
            rotvec = np.array([0.0, 0.0, angle], dtype=np.float32)
        rotmat = R.from_rotvec(rotvec).as_matrix()
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center_pt = (min_bound + max_bound) / 2
        return {
            # Keep legacy key for backward-compat; now it means "the sampled angle".
            "angle_z": float(angle),
            "axis": axis,
            "scaling": float(scaling),
            "rotmat": rotmat,
            "center_pt": center_pt,
        }

    @staticmethod
    def _load_points_colors_normals(
            path: str,
            *,
            return_npz: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional["np.lib.npyio.NpzFile"]],
    ]:
        """Load (points, colors, normals) from either .npz or a point cloud file.

        If return_npz=True and path is a .npz, also returns the loaded NpzFile so
        callers can reuse extra fields (e.g., part_names/part_centers) without a
        second disk read.

        .npz expected keys: 'point', 'color', 'normal'.
        Non-npz: load via Open3D and set normals to zeros.
        """
        if str(path).lower().endswith(".npz"):
            data = np.load(path)
            points = np.asarray(data["point"], dtype=np.float32)
            colors = np.asarray(data["color"], dtype=np.uint8)
            normals = np.asarray(data["normal"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Invalid 'point' shape in {path}: {points.shape}")
            if colors.ndim != 2 or colors.shape[1] != 3:
                raise ValueError(f"Invalid 'color' shape in {path}: {colors.shape}")
            if normals.ndim != 2 or normals.shape[1] != 3:
                raise ValueError(f"Invalid 'normal' shape in {path}: {normals.shape}")
            if not (len(points) == len(colors) == len(normals)):
                raise ValueError(
                    f"Mismatched lengths in {path}: point={len(points)}, color={len(colors)}, normal={len(normals)}"
                )
            if return_npz:
                return points, colors, normals, data
            return points, colors, normals

        pcd = load_o3d_pcd(path)
        points, colors = get_points_and_colors(pcd)
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.uint8)
        normals = np.zeros_like(points, dtype=np.float32)
        if return_npz:
            return points, colors, normals, None
        return points, colors, normals

    @staticmethod
    def _normalize_aabb(points: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float]:
        """AABB normalize to canonical scale.

        Same as:
            center=(max+min)/2
            scale=max(abs((max-min)/2))
            p'=(p-center)/scale

        Returns:
            points_norm: (N,3)
            center: (3,)
            scale: float
        """
        max_ = np.max(points, axis=0)
        min_ = np.min(points, axis=0)
        center = (max_ + min_) / 2.0
        half = (max_ - min_) / 2.0
        scale = float(np.max(np.abs(half)))
        if scale < eps:
            scale = 1.0
        pts = (points - center) / (scale + eps)
        return pts.astype(np.float32), center.astype(np.float32), float(scale)

    @staticmethod
    def _apply_transform(points: np.ndarray, transform: dict) -> np.ndarray:
        rotmat = transform["rotmat"]
        center_pt = transform["center_pt"]
        scaling = transform["scaling"]
        scaled = (points - center_pt) * scaling
        # Use row-vector convention: (N,3) @ (3,3)^T
        return (scaled @ rotmat.T) + center_pt

    @staticmethod
    def _to_np3(x: Any, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must be a 3D vector, got shape {np.asarray(x).shape}")
        return arr

    def _transform_motion(self, motion: dict, transform: dict, motion_type: str) -> dict:
        """Apply the same transform to motion labels.

        Expected keys:
          - axis: (3,)
          - pos: (3,)
          - range: scalar or (2,) (angle range or distance range)
        """
        if motion is None:
            return {}
        motion = deepcopy(motion)

        rotmat = transform["rotmat"]
        center_pt = transform["center_pt"]
        scaling = transform["scaling"]

        # Optional normalization (must be applied BEFORE sampled transform).
        norm = transform.get("norm", None)
        norm_center = None
        norm_scale = None
        if norm is not None:
            norm_center = np.asarray(norm.get("center"), dtype=np.float32).reshape(3)
            norm_scale = float(norm.get("scale"))

        if "axis" in motion and motion["axis"] is not None:
            d = self._to_np3(motion["axis"], "axis")
            # Row-vector convention
            d2 = d @ rotmat.T
            n = np.linalg.norm(d2) + 1e-12
            motion["axis"] = (d2 / n).astype(np.float32)

        if motion_type == "C" and "pos" in motion and motion["pos"] is not None:
            p = self._to_np3(motion["pos"], "pos")
            if norm_center is not None and norm_scale is not None:
                p = (p - norm_center) / (norm_scale + 1e-12)
            # Row-vector convention
            p2 = ((p - center_pt) * scaling) @ rotmat.T + center_pt
            motion["pos"] = p2.astype(np.float32)
        else:
            motion["pos"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # range scaling depends on motion_type:
        # - revolute: angle range, do NOT scale
        # - prismatic: distance range, scale with point cloud scaling
        if "range" in motion and motion["range"] is not None:
            if motion_type not in {"B", "C"}:  # B: prismatic; C: revolute
                raise ValueError(f"Unknown motion_type: {motion_type}. Expected 'C: revolute' or 'B: prismatic'.")
            if motion_type == "B":
                mr = np.asarray(motion["range"], dtype=np.float32)
                # distance range is affected by normalization scale and sampled scaling
                if norm_scale is not None:
                    mr = mr / (norm_scale + 1e-12)
                mr = mr * scaling
                lo = float(np.min(mr))
                hi = float(np.max(mr))
                motion["range"] = np.asarray([lo, hi], dtype=np.float32)
            else:
                # motion_type == "C" (revolute): normalize angles by 2π for stable regression.
                mr = np.asarray(motion["range"], dtype=np.float32)
                mr = mr / (2.0 * math.pi)
                # Canonicalize ordering so range is always [min, max].
                # This keeps labels consistent even if upstream flips axis and negates range.
                lo = float(np.min(mr))
                hi = float(np.max(mr))
                motion["range"] = np.asarray([lo, hi], dtype=np.float32)

        return motion

    @staticmethod
    def _canonicalize_revolute_pivot(
            part_points: np.ndarray,
            axis: np.ndarray,
            pivot: np.ndarray,
    ) -> np.ndarray:
        """Make revolute pivot unique by projecting part centroid onto the rotation axis.

        Given a rotation axis defined by (axis, pivot), any point on that axis is a valid
        pivot. For LLM supervision we canonicalize it so the label is stable:

            pivot_new = pivot + ((centroid - pivot) · axis) * axis

        Args:
            part_points: (N, 3) points of the moving part in the SAME coordinate frame
                as axis/pivot (after all transforms and shared shifts).
            axis: (3,) unit direction vector.
            pivot: (3,) a point on the axis.

        Returns:
            (3,) canonical pivot on the same axis.
        """
        pts = np.asarray(part_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return np.asarray(pivot, dtype=np.float32).reshape(3)

        a = np.asarray(axis, dtype=np.float32).reshape(3)
        an = float(np.linalg.norm(a))
        if not np.isfinite(an) or an < 1e-12:
            return np.asarray(pivot, dtype=np.float32).reshape(3)
        a = a / an

        p0 = np.asarray(pivot, dtype=np.float32).reshape(3)
        c = pts.mean(axis=0)
        t = float(np.dot(c - p0, a))
        new_pivot = (p0 + t * a).astype(np.float32)
        return new_pivot

    def process_messages_physllm(
            self,
            messages: Sequence[Dict[str, str]],
            num_part: int,
            num_object: int,
            num_images: int,
            use_image: bool,
            task_name: str,
    ) -> List[Dict[str, str]]:
        """Replace placeholders for PhysLLM modalities.

        - <part_point_cloud> -> <|part_point_start|><|point_pad|><|part_point_end|>
        - <object_point_cloud> -> <|object_point_start|><|point_pad|><|object_point_end|>
        - <image> -> <|image_start|><|image_pad|><|image_end|> (MVP: no image tokens inserted here)
        """
        messages = deepcopy(messages)
        part_used = 0
        obj_used = 0
        img_used = 0

        for message in messages:
            content = message["content"]
            if not use_image:
                content = content.replace(IMAGE_PLACEHOLDER, "")

            while PART_POINT_CLOUD_PLACEHOLDER in content:
                content = content.replace(
                    PART_POINT_CLOUD_PLACEHOLDER,
                    f"{PART_POINT_S_TOKEN}{self.point_token}{PART_POINT_E_TOKEN}",
                    1,
                )
                part_used += 1

            while OBJECT_POINT_CLOUD_PLACEHOLDER in content:
                content = content.replace(
                    OBJECT_POINT_CLOUD_PLACEHOLDER,
                    f"{OBJECT_POINT_S_TOKEN}{self.point_token}{OBJECT_POINT_E_TOKEN}",
                    1,
                )
                obj_used += 1

            while IMAGE_PLACEHOLDER in content:
                # If this sample has no image, drop the placeholder entirely.
                # if num_images == 0:
                #     content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                # else:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{IMAGE_S_TOKEN}{self.image_token}{IMAGE_E_TOKEN}",
                    1,
                )
                img_used += 1

            message["content"] = content

        if task_name not in {"object_level"} and part_used != num_part:
            raise ValueError(
                f"Expected {num_part} {PART_POINT_CLOUD_PLACEHOLDER} placeholders, got {part_used}."
            )
        if obj_used != num_object:
            raise ValueError(
                f"Expected {num_object} {OBJECT_POINT_CLOUD_PLACEHOLDER} placeholders, got {obj_used}."
            )
        if use_image:
            if num_images != 0 and img_used != num_images:
                raise ValueError(
                    f"Expected {num_images} {IMAGE_PLACEHOLDER} placeholders, got {img_used}."
                )
        return messages

    def get_mm_inputs(
            self,
            part_point_clouds: Sequence["PointCloudInput"],
            object_point_clouds: Sequence["PointCloudInput"],
            images: Sequence[Union[str, Sequence[str]]],
            batch_prompts: Sequence[List[dict]],
            motions: Sequence[dict],
            task_name: str,
            use_image: bool,
    ) -> Dict[str, Union[List[dict], torch.Tensor, List[List[str]], Dict[str, torch.Tensor]]]:
        """Build PhysLLM multimodal inputs.

        Args:
            part_point_clouds/object_point_clouds: list of pcd paths (one per sample in MVP)
            images: list of image path(s) per sample
            batch_prompts: list of messages per sample
            motions: list of motion dict per sample
        """
        input_dict: Dict[str, Any] = {
            "part_point_clouds": None,
            "object_point_clouds": None,
            "images": None,
            "motion_labels": None,
            "centroids": None,
        }

        if task_name == "object_level":
            part_point_clouds = object_point_clouds

        if not (
                len(batch_prompts)
                == len(part_point_clouds)
                == len(object_point_clouds)
                == len(motions)
                == len(images)
        ):
            raise ValueError(
                "Batch size mismatch among prompts/part/object/motions/images: "
                f"{len(batch_prompts)}/{len(part_point_clouds)}/{len(object_point_clouds)}/{len(motions)}/{len(images)}"
            )

        part_data = []
        obj_data = []
        motion_out: Dict[str, List[np.ndarray]] = {
            "axis": [],
            "pos": [],
            "range": [],
        }
        centroids_out: List[np.ndarray] = []
        processed_messages = []
        images_out: List[List[str]] = []
        # Per-sample motion type code: 0 -> B (prismatic/translation), 1 -> C (revolute/rotation)
        motion_type_codes: List[int] = []
        is_motion_task = task_name in {"motion"}
        is_group_task = task_name in {"group"}
        for i in range(len(batch_prompts)):
            part_paths = self._as_list(part_point_clouds[i])
            obj_paths = self._as_list(object_point_clouds[i])
            img_paths = self._as_list(images[i])
            motion_i = motions[i] or {}
            motion_type = None
            parsed_motion: Optional[dict] = None
            parsed_assistant_idx: Optional[int] = None

            if is_motion_task:
                # If spherical-axis mode is enabled, force the user prompt to use the
                # spherical-axis schema so assistant JSON and label parsing stay consistent.
                if self.use_spherical_axis:
                    for msg in batch_prompts[i]:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            msg["content"] = MotionTask.DEFAULT_PROMPT3
                            break

                # Motion task: always parse motion labels from assistant JSON in batch_prompts.
                # This keeps labels consistent with the prompt/label format.
                try:
                    for j, msg in enumerate(batch_prompts[i]):
                        if not isinstance(msg, dict):
                            continue
                        if msg.get("role") != "assistant":
                            continue
                        content = msg.get("content")
                        if not isinstance(content, str) or not content.strip():
                            continue
                        parsed_motion = json.loads(content)
                        parsed_assistant_idx = j
                        break
                except Exception:
                    parsed_motion = None
                    parsed_assistant_idx = None

                if not isinstance(parsed_motion, dict):
                    raise ValueError("Motion task requires assistant JSON motion label in batch_prompts.")

                mt = parsed_motion.get("motion_type")
                if mt not in {"B", "C"}:
                    raise ValueError(f"Invalid motion_type in assistant JSON: {mt}. Expected 'B' or 'C'.")

                motion_i = {
                    mt: {
                        "axis": parsed_motion.get("axis"),
                        # internal motion label uses key "pos"
                        "pos": parsed_motion.get("pivot"),
                        "range": parsed_motion.get("range"),
                    }
                }

                # motion_i is expected to be like: {"B": {axis,pos,range}} or {"C": {...}}
                # where B: prismatic (translation), C: revolute (rotation)
                if not isinstance(motion_i, dict):
                    raise ValueError(
                        f"Motion info must be a dict like {{'B':{{...}}}}/{{'C':{{...}}}}, got {type(motion_i)}"
                    )

                keys = [
                    k
                    for k in motion_i.keys()
                    if k is not None
                       and str(k).strip() != "dependency"
                       and isinstance(motion_i.get(k), dict)
                       and motion_i.get(k).get("axis") is not None
                ]
                if len(keys) == 0:
                    raise ValueError(
                        f"Motion info must contain at least one motion type key ('B' or 'C'), got keys={keys}"
                    )

                # Allow multiple keys but only use the first one (consistent with dataset heuristic).
                motion_type = str(keys[0]).strip()
                if motion_type not in {"B", "C"}:
                    raise ValueError(
                        f"Unknown motion_type: {motion_type}. Expected 'B' (translation) or 'C' (rotation)."
                    )

                motion_type_codes.append(0 if motion_type == "B" else 1)

                motion_i = motion_i[motion_type] or {}
                if (
                        "axis" not in motion_i
                        or "pos" not in motion_i
                        or "range" not in motion_i
                ):
                    raise ValueError(
                        "Missing specific fields (axis, pos or range) in motion info!"
                    )
            else:
                # Non-motion tasks: ignore motion labels entirely.
                motion_i = {}
            if not is_motion_task:
                # keep placeholder for shape consistency (will not be used)
                motion_type_codes.append(-1)

            if len(part_paths) != 1 or len(obj_paths) != 1:
                raise ValueError(
                    "PhysMeshLLMPlugin MVP expects exactly one part and one object point cloud path per sample. "
                    f"Got part={len(part_paths)}, object={len(obj_paths)}"
                )

            # load object first to sample transform
            # (Group task) also reuse the same loaded npz to get part metadata.
            obj_points, obj_colors, obj_normals, obj_npz = self._load_points_colors_normals(
                obj_paths[0],
                return_npz=True,
            )
            part_points, part_colors, part_normals = self._load_points_colors_normals(part_paths[0])

            # (Group task) Load part centers/names from the same object npz before any transforms.
            part_names = None
            part_centers = None
            if is_group_task and obj_npz is not None:
                try:
                    if "part_names" in obj_npz and "part_centers" in obj_npz:
                        part_names = list(obj_npz["part_names"])
                        part_centers = np.asarray(obj_npz["part_centers"], dtype=np.float32)
                except Exception:
                    part_names, part_centers = None, None

            # 0) Normalize FIRST (required): unify all mesh scales.
            # Use object cloud AABB for normalization and apply to both object+part.
            obj_points, norm_center, norm_scale = self._normalize_aabb(obj_points)
            part_points = ((part_points - norm_center) / (norm_scale + 1e-12)).astype(np.float32)
            if is_group_task and part_centers is not None:
                # centers are points -> apply the same normalization
                part_centers = ((part_centers - norm_center) / (norm_scale + 1e-12)).astype(np.float32)

            # normals are direction vectors, no need to center/scale
            obj_normals = np.asarray(obj_normals, dtype=np.float32)
            part_normals = np.asarray(part_normals, dtype=np.float32)

            if self.do_augmentation:
                obj_aug = {"name": "pcd", "coord": obj_points, "color": obj_colors, "normal": obj_normals}
                obj_aug = self.augmentation(obj_aug)
                obj_points, obj_colors, obj_normals = obj_aug["coord"], obj_aug["color"], obj_aug["normal"]

                part_aug = {"name": "pcd", "coord": part_points, "color": part_colors, "normal": part_normals}
                part_aug = self.augmentation(part_aug)
                part_points, part_colors, part_normals = part_aug["coord"], part_aug["color"], part_aug["normal"]

            transform = self._sample_transform(obj_points)
            transform["norm"] = {"center": norm_center, "scale": float(norm_scale)}
            obj_points_t = self._apply_transform(obj_points, transform)
            part_points_t = self._apply_transform(part_points, transform)
            if is_group_task and part_centers is not None:
                part_centers_t = self._apply_transform(part_centers, transform)

            # rotate normals (do not apply translation/scaling)
            rot = np.asarray(transform["rotmat"], dtype=np.float32)
            obj_normals_t = (obj_normals @ rot.T).astype(np.float32)
            part_normals_t = (part_normals @ rot.T).astype(np.float32)

            # store min_bound for normalization shift (shared)
            # NOTE: disabled. points are already AABB-normalized earlier, and a global
            # shift to make all coords positive is not required for the model.
            # Keeping this block commented for reproducibility.
            min_bound = np.minimum(obj_points_t.min(axis=0), part_points_t.min(axis=0))
            obj_points_t = obj_points_t - min_bound
            part_points_t = part_points_t - min_bound
            if is_group_task and part_centers is not None:
                part_centers_t = part_centers_t - min_bound

            # 2) Apply a SHARED CenterShift at the end (after sampled transform), so both
            # object and part remain aligned and the final coordinate frame is centered.
            # This mirrors pcd.transform.CenterShift (xy center + z_min).
            # shared_shift = self._compute_center_shift_from_object(obj_points_t, apply_z=True)
            # obj_points_t = (obj_points_t - shared_shift).astype(np.float32)
            # part_points_t = (part_points_t - shared_shift).astype(np.float32)
            # if is_group_task and part_centers is not None:
            #     part_centers_t = (part_centers_t - shared_shift).astype(np.float32)

            obj_pc = {"name": "pcd", "coord": obj_points_t, "color": obj_colors, "normal": obj_normals_t}
            part_pc = {"name": "pcd", "coord": part_points_t, "color": part_colors, "normal": part_normals_t}
            obj_data.append(obj_pc)
            part_data.append(part_pc)

            if is_motion_task:
                motion_t = self._transform_motion(motion_i, transform, motion_type)
                # also apply the same min_bound shift to axis_position
                # NOTE: disabled together with the point cloud min_bound shift above.
                if "pos" in motion_t and motion_t["pos"] is not None:
                    motion_t["pos"] = (
                            np.asarray(motion_t["pos"], dtype=np.float32)
                            - min_bound.astype(np.float32)
                    )

                # Keep motion labels in the same final coordinate frame as point clouds.
                # axis is a direction -> unaffected by translation.
                # pos is a point -> must subtract the same shared_shift.
                # if motion_type == "C" and "pos" in motion_t and motion_t["pos"] is not None:
                #     motion_t["pos"] = (
                #             np.asarray(motion_t["pos"], dtype=np.float32)
                #             - shared_shift.astype(np.float32)
                #     )

                # Canonicalize revolute pivot for label uniqueness:
                # project the part centroid onto the (axis, pivot) line.
                # if motion_type == "C":
                #     try:
                #         if motion_t.get("axis") is not None and motion_t.get("pos") is not None:
                #             motion_t["pos"] = self._canonicalize_revolute_pivot(
                #                 part_points=part_points_t,
                #                 axis=np.asarray(motion_t["axis"], dtype=np.float32).reshape(3),
                #                 pivot=np.asarray(motion_t["pos"], dtype=np.float32).reshape(3),
                #             )
                #     except Exception:
                #         # Keep original pivot if canonicalization fails.
                #         pass

                # Canonicalize prismatic pivot for label uniqueness:
                # use the moving part centroid directly.
                if motion_type == "B":
                    try:
                        pts = np.asarray(part_points_t, dtype=np.float32)
                        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                            motion_t["pos"] = pts.mean(axis=0).astype(np.float32)
                    except Exception:
                        pass

                # Write transformed motion back into the assistant JSON (prompt label),
                # so downstream sees labels in the same final coordinate frame as point clouds.
                if parsed_motion is not None and parsed_assistant_idx is not None:
                    # Quantize to 4 decimals (stored as numbers).
                    axis_vec = np.asarray(motion_t.get("axis"), dtype=np.float32).reshape(3)
                    if self.use_spherical_axis:
                        flipped = self._should_flip_axis_canonical(axis_vec)
                        theta_deg, phi_deg = self._vector_to_theta_phi_deg(axis_vec)
                        theta_i = int(round(theta_deg))
                        phi_i = int(round(phi_deg))
                        # clamp to conventional ranges
                        if theta_i < 0:
                            theta_i = 0
                        elif theta_i > 180:
                            theta_i = 180
                        # wrap phi into [0, 359]
                        phi_i = phi_i % 360
                        parsed_motion["axis"] = {
                            "theta": theta_i,
                            "phi": phi_i,
                        }

                        # If we canonicalize by flipping axis direction, signed range
                        # must flip as well to preserve motion direction semantics.
                        if flipped:
                            rr0 = np.asarray(motion_t.get("range"), dtype=np.float32).reshape(-1)
                            if rr0.size == 2:
                                rr0 = -rr0
                                lo0 = float(np.min(rr0))
                                hi0 = float(np.max(rr0))
                                motion_t["range"] = np.asarray([lo0, hi0], dtype=np.float32)
                    else:
                        parsed_motion["axis"] = [float(f"{float(x):.4f}") for x in axis_vec]
                    parsed_motion["pivot"] = [float(f"{float(x):.4f}") for x in
                                              np.asarray(motion_t.get("pos"), dtype=np.float32).reshape(3)]
                    rr = np.asarray(motion_t.get("range"), dtype=np.float32).reshape(-1)
                    if rr.size != 2:
                        raise ValueError(f"Motion range must be 2D [min,max], got shape={rr.shape}.")
                    parsed_motion["range"] = [float(f"{float(rr[0]):.4f}"), float(f"{float(rr[1]):.4f}")]

                    # Replace assistant content with the updated JSON string.
                    batch_prompts[i][parsed_assistant_idx]["content"] = json.dumps(parsed_motion, ensure_ascii=False)

                # Motion task requires these fields.
                if (
                        motion_t.get("axis") is None
                        or motion_t.get("pos") is None
                        or motion_t.get("range") is None
                ):
                    print(f"Original motion labels for sample {i}: {motion_i}")
                    print(f"Invalid motion labels after transform for sample {i}: {motion_t}")
                    raise ValueError(
                        "Motion task requires axis/pos/range in _motions."
                    )

                axis_dir = np.asarray(motion_t["axis"], dtype=np.float32)
                axis_pos = np.asarray(motion_t["pos"], dtype=np.float32)
                mr_arr = np.asarray(motion_t["range"], dtype=np.float32)
                motion_out["axis"].append(axis_dir)
                motion_out["pos"].append(axis_pos)
                motion_out["range"].append(mr_arr)

                # Store centroid (center of mass proxy) in the SAME final coordinate frame.
                # This is used for physics-inspired joint position loss.
                try:
                    pts = np.asarray(part_points_t, dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                        centroids_out.append(pts.mean(axis=0).astype(np.float32))
                    else:
                        centroids_out.append(np.zeros((3,), dtype=np.float32))
                except Exception:
                    centroids_out.append(np.zeros((3,), dtype=np.float32))

            processed_messages.append(
                self.process_messages_physllm(
                    self._inject_group_part_lists(
                        batch_prompts[i],
                        task_name=task_name,
                        part_names=part_names,
                        part_centers_t=part_centers_t if is_group_task and part_centers is not None else None,
                    ),
                    num_part=len(part_paths),
                    num_object=len(obj_paths),
                    num_images=len(img_paths),
                    use_image=use_image,
                    task_name=task_name
                )
            )
            images_out.append(img_paths)

        input_dict["messages"] = processed_messages
        input_dict["motion_types"] = torch.as_tensor(motion_type_codes, dtype=torch.long)

        # Compute shared grid_min_coord from each object point cloud so that
        # part and object grid_coords are in the same reference frame.
        # This is critical for Fourier positional encoding consistency.
        shared_grid_min_coords = (
            [self._compute_shared_grid_min_coord(obj_pc) for obj_pc in obj_data]
            if self.share_grid_origin
            else None
        )

        input_dict["part_point_clouds"] = self._regularize_point_clouds(
            part_data, shared_grid_min_coords=shared_grid_min_coords
        )
        input_dict["object_point_clouds"] = self._regularize_point_clouds(
            obj_data, shared_grid_min_coords=shared_grid_min_coords
        )
        # images: return normalized tensor for model.forward(images=...)
        input_dict["images"] = self._regularize_images(images_out, use_image)

        if is_motion_task:
            # convert motion labels to a single tensor to satisfy dataloader constraints
            # layout: [axis(3), pos(3), range(2)] -> (B, 8)
            axis_t = torch.as_tensor(np.stack(motion_out["axis"], axis=0), dtype=torch.float32)
            pos_t = torch.as_tensor(np.stack(motion_out["pos"], axis=0), dtype=torch.float32)
            range_t = torch.as_tensor(np.stack(motion_out["range"], axis=0), dtype=torch.float32)
            input_dict["motion_labels"] = torch.cat([axis_t, pos_t, range_t], dim=1)
            input_dict["centroids"] = torch.as_tensor(np.stack(centroids_out, axis=0), dtype=torch.float32)
        else:
            input_dict["motion_labels"] = None
            input_dict["centroids"] = None

        return input_dict

    def _inject_group_part_lists(
            self,
            messages: List[dict],
            *,
            task_name: str,
            part_names: Optional[List[Any]],
            part_centers_t: Optional[np.ndarray],
    ) -> List[dict]:
        """If task is group, inject {{PART_LIST}} into the user prompt.

        Builds part list from:
          - object npz: part_names (list) and part_centers (N,3)

                Note:
                    - We keep ids as strings.
        """
        if task_name != "group":
            return messages

        # Only modify user message contents.
        out = deepcopy(messages)

        # Build list items in the order of part_names/part_centers (1-1).
        parts_out: List[dict] = []
        if part_names is not None and part_centers_t is not None:
            try:
                n = min(len(part_names), int(part_centers_t.shape[0]))
            except Exception:
                n = 0
            for idx in range(n):
                pid = str(part_names[idx])
                center = part_centers_t[idx]
                parts_out.append(
                    {
                        "id": pid,
                        "position": [
                            float(f"{float(center[0]):.4f}"),
                            float(f"{float(center[1]):.4f}"),
                            float(f"{float(center[2]):.4f}"),
                        ],
                    }
                )
        part_list_str = json.dumps(parts_out, ensure_ascii=False)

        for m in out:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if "{{PART_LIST}}" in content:
                content = content.replace("{{PART_LIST}}", part_list_str)
            m["content"] = content

        return out

    def _validate_input(
            self,
            point_clouds: Sequence["PointCloudInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(point_clouds) != 0 and self.point_token is None:
            raise ValueError(
                "This model does not support point cloud input. Please check whether the correct `template` is used."
            )

    def process_token_ids(
            self,
            input_ids: List[int],
            labels: Optional[List[int]],
            point_clouds: Sequence["PointCloudInput"],
            tokenizer: "PreTrainedTokenizer",
    ) -> Tuple[List[int], Optional[List[int]]]:
        self._validate_input(point_clouds)
        return input_ids, labels


# Backward-compatible alias. Legacy alias retained from the codebase refactor.
PhysMeshLLMPlugin = UniPhysGenPlugin


def get_mm_plugin(
        point_token: str = "<|point_pad|>",
        image_token: str = "<|image_pad|>",
        **kwargs,
) -> "PhysMeshLLMPlugin":
    return PhysMeshLLMPlugin(point_token=point_token, image_token=image_token, **kwargs)
