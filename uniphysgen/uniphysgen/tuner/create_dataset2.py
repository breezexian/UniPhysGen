import json
import os

import numpy as np
import trimesh
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from subprocess import call, check_call


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

    # 防御：部分资产会有非法 UV（NaN/inf）或 barycentric 数值问题。
    # 这里统一把非法值替换到 [0,1] 范围内，避免 int64 cast 溢出导致 -9223372036854775808。
    uv_s = np.nan_to_num(uv_s, nan=0.0, posinf=1.0, neginf=0.0)

    # UV -> 像素坐标（trimesh/大多数 UV：v 向上，需要翻转到图像坐标）
    img = np.asarray(image)
    h, w = img.shape[0], img.shape[1]
    u = np.clip(uv_s[:, 0], 0.0, 1.0)
    v = np.clip(uv_s[:, 1], 0.0, 1.0)

    # x = (u * (w - 1)).astype(np.int64)
    # y = ((1.0 - v) * (h - 1)).astype(np.int64)
    x = np.floor(u * (w - 1)).astype(np.int64)
    y = np.floor((1.0 - v) * (h - 1)).astype(np.int64)
    # 双保险：clip 到合法索引
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

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
    # try:
    sample_pointcloud_with_color(full_mesh, sample_points, save_ply_path=os.path.join(ply_dir, "model.ply"),
                                 save_npz_path=os.path.join(npz_dir, "model.npz"))
    # except Exception as e:
    #     print(e)
    #     print(full_model_pth)
    #     assert 1==2
    # collect part centers and names for post-processing into model.npz
    part_names: list[str] = []
    part_centers: list[np.ndarray] = []
    obj_files = sorted(obj_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    for file_pth in obj_files:
        part_name = os.path.basename(file_pth).split(".")[0]
        part_mesh = trimesh.load(file_pth, force='mesh')
        part_area = part_mesh.area
        rate = part_area / full_area
        part_points_num = int(max(10000, rate * sample_points))
        # try:
        part_points, part_colors, part_normals = sample_pointcloud_with_color(
            part_mesh,
            part_points_num,
            save_ply_path=os.path.join(ply_dir, f"{part_name}.ply"),
            save_npz_path=os.path.join(npz_dir, f"{part_name}.npz"),
        )
        # except Exception as e:
        #     print(e)
        #     print(file_pth)
        #     assert 1==2
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
    # except Exception as e:
    #     print(e)
    #     print(entity_dir)
    #     return
    return

if __name__ == '__main__':

    # pth = "/seaweedfs/xianzi/data/sample_plys/abo/0/B00BBDF500/npzs/model.npz"
    # a = dict(np.load(pth))
    # a = 1
    res_roots = {
        # "abo": "/seaweedfs/xianzi/data/abo-3dmodels_res",
        # "partnet": "/seaweedfs/xianzi/data/",
        # "objaverse": "/seaweedfs/xianzi/data/phys_objaverse_sketchfab_trellis_res",
        "hssd": "/seaweedfs/xianzi/data/hssd_res",
        # "future": "/seaweedfs/xianzi/data/future_res"
    }

    save_dir = "/seaweedfs/xianzi/data/sample_plys"
    for dataset in res_roots:
        res_root = res_roots[dataset]
        type_names = os.listdir(res_root)
        if "partnet" in dataset:
            type_names = ["partnet_texture_phys"]
        for type_name in tqdm(type_names):
            # if type_name != "0":
            #     continue
            res_type_dir = os.path.join(res_root, type_name)
            entities = os.listdir(res_type_dir)
            # for entity in tqdm(entities):
            #     cur_entity_dir = os.path.join(res_type_dir, entity)
            #     entity_save_dir = os.path.join(save_dir, dataset, type_name, entity)
            #     generate_ply_per_entity(cur_entity_dir, entity_save_dir)
            futures = []
            results = []
            with ProcessPoolExecutor(max_workers=10) as executor:
                for entity in tqdm(entities):
                    cur_entity_dir = os.path.join(res_type_dir, entity)
                    entity_save_dir = os.path.join(save_dir, dataset, type_name, entity)
                    futures.append(executor.submit(generate_ply_per_entity, cur_entity_dir, entity_save_dir))
                for future in as_completed(futures):
                    results.append(future.result())
                    print(future.result())
