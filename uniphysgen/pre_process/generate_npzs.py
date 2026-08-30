"""Generate UniPhysGen-ready point clouds from object-part meshes."""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from subprocess import call

import numpy as np
import trimesh
from tqdm import tqdm


_PART_OBJ_PATTERN = re.compile(r"^(?:part_?)?(\d+)$", flags=re.IGNORECASE)


def _part_id_from_obj(file_path: str) -> str:
    """Return the numeric ID encoded by ``part_<id>.obj``."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = _PART_OBJ_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Invalid part OBJ name: {os.path.basename(file_path)!r}. "
            "Expected 'part_<number>.obj'."
        )
    return match.group(1)


def _sample_colors_from_material(mesh, n_points):
    visual = mesh.visual
    mat = getattr(visual, "material", None)
    if mat is None:
        return None

    # trimesh 的 Kd
    kd = getattr(mat, "diffuse", None)
    if kd is None:
        return None

    kd = np.array(kd)

    # # 有些是 0~1，需要转 0~255
    # if kd.max() <= 1.0:
    #     kd = (kd * 255).astype(np.uint8)
    # else:
    kd = kd.astype(np.uint8)

    return np.tile(kd[:3], (n_points, 1))


def _sample_colors_from_texture(mesh: trimesh.Trimesh, face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray | None:
    """
    从纹理（UV + image）里采样颜色。成功返回 (N,3) uint8，否则返回 None。
    """
    visual = mesh.visual
    if not hasattr(visual, "uv") or visual.uv is None:
        return None
    if getattr(visual, "material", None) is None:
        return None
    image = getattr(visual.material, "image", None)
    if image is None:
        return None

    image = image.convert("RGB")

    uv = np.asarray(visual.uv)  # (n_verts, 2) or (n_uv,2)
    faces = np.asarray(mesh.faces)

    # 将每个采样点落在的三角形顶点 UV 做重心插值
    tri = faces[face_idx]  # (N, 3)
    uv_tri = uv[tri]  # (N, 3, 2)
    uv_s = (uv_tri[:, 0, :] * bary[:, 0:1] +
            uv_tri[:, 1, :] * bary[:, 1:2] +
            uv_tri[:, 2, :] * bary[:, 2:3])  # (N, 2)
    uv_s = np.nan_to_num(uv_s, nan=0.0, posinf=1.0, neginf=0.0)
    # UV -> 像素坐标（trimesh/大多数 UV：v 向上，需要翻转到图像坐标）
    img = np.asarray(image)
    h, w = img.shape[0], img.shape[1]
    u = np.clip(uv_s[:, 0], 0.0, 1.0)
    v = np.clip(uv_s[:, 1], 0.0, 1.0)
    x = (u * (w - 1)).astype(np.int64)
    y = ((1.0 - v) * (h - 1)).astype(np.int64)

    rgb = img[y, x]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :3]
    return rgb.astype(np.uint8)


def _sample_colors_from_visual(mesh: trimesh.Trimesh, face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """
    fallback：face_colors / vertex_colors
    """
    visual = mesh.visual

    # face_colors: (F,4) or (F,3)
    fc = getattr(visual, "face_colors", None)
    if fc is not None and len(fc) == len(mesh.faces):
        rgb = np.asarray(fc)[face_idx]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :3]
        return rgb.astype(np.uint8)

    # vertex_colors: (V,4) or (V,3)
    vc = getattr(visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        tri = np.asarray(mesh.faces)[face_idx]  # (N,3)
        vcol = np.asarray(vc)[tri]  # (N,3,3/4)
        if vcol.shape[-1] == 4:
            vcol = vcol[:, :, :3]
        rgb = (vcol[:, 0, :] * bary[:, 0:1] +
               vcol[:, 1, :] * bary[:, 1:2] +
               vcol[:, 2, :] * bary[:, 2:3])
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # 都没有的话给白色
    return np.full((len(face_idx), 3), 255, dtype=np.uint8)


def sample_pointcloud_with_color(
        mesh: trimesh.Trimesh,
        n_points: int,
        seed: int | None = 42,
        save_ply_path: str | None = None,
        save_npz_path: str | None = None,
):
    """
    返回:
      points: (N,3) float32
      colors: (N,3) uint8
            normals: (N,3) float32
    """
    # 采样点 + face_idx
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)

    # 同时得到这些点在三角形内的重心坐标，用于插值颜色/UV
    faces = np.asarray(mesh.faces)
    v = np.asarray(mesh.vertices)
    tri = faces[face_idx]  # (N,3)
    tri_v = v[tri]  # (N,3,3)
    bary = trimesh.triangles.points_to_barycentric(triangles=tri_v, points=points)  # (N,3)

    # 颜色：优先纹理，否则用 visual 颜色
    colors = _sample_colors_from_texture(mesh, face_idx, bary)

    if colors is None:
        colors = _sample_colors_from_material(mesh, len(face_idx))

    if colors is None:
        colors = _sample_colors_from_visual(mesh, face_idx, bary)

    # 法向量：用面法向量作为点法向量（每个采样点落在一个三角面上）
    face_normals = np.asarray(mesh.face_normals)
    if len(face_normals) != len(mesh.faces):
        # 保险：确保 face_normals 可用
        mesh.rezero()
        mesh.fix_normals()
        face_normals = np.asarray(mesh.face_normals)
    normals = face_normals[face_idx].astype(np.float32)
    # 归一化
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(nrm, 1e-12, None)

    points = points.astype(np.float32)
    colors = colors.astype(np.uint8)

    if save_ply_path is not None:
        pc = trimesh.points.PointCloud(points, colors=colors)
        pc.export(save_ply_path)

    if save_npz_path is not None:
        # 以字典形式保存
        np.savez_compressed(save_npz_path, point=points, color=colors, normal=normals)

    return points, colors, normals


def get_useful_objs(entity_dir):
    obj_dir = os.path.join(entity_dir, "parts")
    if os.path.exists(obj_dir):
        files = os.listdir(obj_dir)
        objs = []
        for file in files:
            if not file.endswith(".obj"):
                continue
            objs.append(os.path.join(obj_dir, file))
        return obj_dir, objs

    # Backward compatibility with the internal dataset layout.
    obj_dir = os.path.join(entity_dir, "merge_objs")
    glb_dir = os.path.join(entity_dir, "glbs")  # 没有处理完整
    if os.path.exists(glb_dir):
        return None, None
    if not os.path.exists(obj_dir):
        obj_dir = os.path.join(entity_dir, "objs")
    if not os.path.exists(obj_dir):
        return None, None
    files = os.listdir(obj_dir)
    objs = []
    files = os.listdir(obj_dir)
    for file in files:
        if not file.endswith(".obj"):
            continue
        label = file.split(".")[0]
        objs.append(os.path.join(obj_dir, file))
    return obj_dir, objs


def generate_ply_per_entity(entity_dir, save_dir, sample_points=100000):
    try:
        obj_dir, obj_files = get_useful_objs(entity_dir)
        if obj_dir is None:
            return
        ply_dir = os.path.join(entity_dir, "plys")
        npz_dir = os.path.join(save_dir, "npzs")
        full_model_dir = os.path.join(entity_dir, "full_model")
        if os.path.exists(npz_dir) and len(os.listdir(npz_dir)) == len(obj_files) + 1:
            return
        if os.path.exists(save_dir):
            cmd = f"rm -rf {save_dir} {full_model_dir}"
            call(cmd, shell=True)
            print(cmd)
        # try:
        meshes = []
        full_model_pth = os.path.join(full_model_dir, "model.obj")
        if not os.path.exists(full_model_pth):
            for file_pth in obj_files:
                meshes.append(trimesh.load(file_pth, force='mesh'))
            full_mesh = trimesh.util.concatenate(meshes)
            os.makedirs(full_model_dir, exist_ok=True)
            full_mesh.export(full_model_pth)
        full_mesh = trimesh.load(full_model_pth, force='mesh')
        full_area = full_mesh.area
        os.makedirs(ply_dir, exist_ok=True)
        os.makedirs(npz_dir, exist_ok=True)
        sample_pointcloud_with_color(full_mesh, sample_points, save_ply_path=os.path.join(ply_dir, "model.ply"),
                                     save_npz_path=os.path.join(npz_dir, "model.npz"))
        # collect part centers and names for post-processing into model.npz
        part_names: list[str] = []
        part_centers: list[np.ndarray] = []
        obj_files = sorted(obj_files, key=lambda x: int(_part_id_from_obj(x)))
        for file_pth in obj_files:
            part_name = _part_id_from_obj(file_pth)
            part_mesh = trimesh.load(file_pth, force='mesh')
            part_area = part_mesh.area
            rate = part_area / full_area
            part_points_num = int(max(10000, rate * sample_points))
            part_points, part_colors, part_normals = sample_pointcloud_with_color(
                part_mesh,
                part_points_num,
                save_ply_path=os.path.join(ply_dir, f"{part_name}.ply"),
                save_npz_path=os.path.join(npz_dir, f"{part_name}.npz"),
            )
            # part center: use sampled point cloud centroid
            center = np.mean(part_points, axis=0).astype(np.float32)
            part_names.append(part_name)
            part_centers.append(center)

        # write part metadata back into model.npz
        model_npz_path = os.path.join(npz_dir, "model.npz")
        model_data = dict(np.load(model_npz_path))
        # store: names (string array) and centers (K,3)
        model_data["part_names"] = np.asarray(part_names, dtype=np.str_)
        model_data["part_centers"] = np.stack(part_centers, axis=0).astype(np.float32) if len(
            part_centers) > 0 else np.zeros((0, 3), dtype=np.float32)
        np.savez_compressed(model_npz_path, **model_data)
    except Exception as e:
        print(e)
        print(entity_dir)
        return


def _find_entities(data_root: str) -> list[str]:
    """Find sample directories that contain at least one part OBJ."""
    entities = []
    for current_dir, dir_names, _ in os.walk(data_root):
        if "parts" not in dir_names:
            continue
        parts_dir = os.path.join(current_dir, "parts")
        if any(name.lower().endswith(".obj") for name in os.listdir(parts_dir)):
            entities.append(current_dir)
        # A sample's parts are inputs, never nested samples.
        dir_names.remove("parts")
    return sorted(entities)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert every sample below DATA_ROOT from parts/part_<id>.obj "
            "meshes to UniPhysGen .npz point clouds."
        )
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory of UniPhys-Bench, UniPhys-40K, or a compatible dataset.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for generated point clouds; input subdirectories are preserved.",
    )
    parser.add_argument(
        "--sample_points",
        type=_positive_int,
        default=100000,
        help="Number of points sampled for each complete object (default: 100000).",
    )
    parser.add_argument(
        "--num_workers",
        type=_positive_int,
        default=10,
        help="Number of samples processed in parallel (default: 10).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isdir(data_root):
        raise SystemExit(f"Data root does not exist or is not a directory: {data_root}")

    entities = _find_entities(data_root)
    if not entities:
        raise SystemExit(
            f"No samples containing parts/part_<id>.obj were found below: {data_root}"
        )

    os.makedirs(output_dir, exist_ok=True)
    workers = min(args.num_workers, len(entities))
    print(f"Found {len(entities)} samples. Writing point clouds to: {output_dir}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for entity_dir in entities:
            relative_dir = os.path.relpath(entity_dir, data_root)
            entity_output_dir = os.path.join(output_dir, relative_dir)
            futures.append(
                executor.submit(
                    generate_ply_per_entity,
                    entity_dir,
                    entity_output_dir,
                    args.sample_points,
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating NPZs",
        ):
            future.result()

    print(f"Finished preprocessing {len(entities)} samples.")


if __name__ == "__main__":
    """
    python pre_process/generate_npzs.py --data_root /data-koolab-nas/xianzi/uniphys-40k_release --output_dir /data-koolab-nas/xianzi/data/UniPhys-40K-processed/npzs
    
    CUDA_VISIBLE_DEVICES=0 python inference_batch_kinematic_parameters.py \
  --model_path /data-koolab-nas/xianzi/code/physmeshllm/physmeshllm_base-1.7B-motion_full_spherical_0_2_share_newxyz \
  --input_json /data-koolab-nas/xianzi/data/UniPhys-Bench-processed/manifests/kinematic_parameters.json \
  --data_root /data-koolab-nas/xianzi/data/UniPhys-Bench-processed/npzs \
  --spherical_axis \
  --output outputs/UniPhys-Bench/kinematic_parameters.json
    
    """
    main()
