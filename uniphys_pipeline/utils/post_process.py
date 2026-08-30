from copy import deepcopy

import trimesh
import numpy as np
from scipy.spatial import KDTree
import os
import json

from .mesh_handle import read_mesh


def get_face_areas(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 两条边向量
    e1 = v1 - v0
    e2 = v2 - v0

    # 三角形面积 = 0.5 * |e1 × e2|
    cross_prod = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    return areas


def get_collision_mesh(mesh):
    try:
        mc = mesh.convex_hull.copy()  # 避免引用原始buffer
        return mc
    except Exception as e:
        print(f"Convex hull generation failed: {e}")
        return mesh.copy()


def cat_aabb_mesh_volume(mesh):
    bounds = mesh.bounds
    aabb_volume = np.prod(bounds[1] - bounds[0])
    return aabb_volume


def cat_approximate_mesh_volume(mesh, n_samples=2000):
    # 在mesh1的AABB里面采样点
    bounds = mesh.bounds
    # print(bounds)
    samples = np.random.uniform(bounds[0], bounds[1], size=(n_samples, 3))

    # 判断哪些点同时在两个mesh里面
    inside = mesh.contains(samples)

    # 比例 * AABB体积 ≈ mesh体积
    aabb_volume = np.prod(bounds[1] - bounds[0])
    # print(aabb_volume)
    return inside.sum() / n_samples * aabb_volume


def cat_convex_hull_volume(mesh):
    colli_mesh = get_collision_mesh(mesh)
    # colli_mesh.show()
    return colli_mesh.volume


def filter_invalid_part(mesh):
    bbox = mesh.bounding_box.extents
    A = mesh.area
    shortest_edge = bbox.min()
    if shortest_edge < 1e-6 or A < 1e-10:
        return True
    return False


def edge_min_max_rate(mesh, threshold=100, eps=1e-6):
    """
    判断一个mesh是否为细长形状：
    当最长边 / 两条较短边 > threshold 时返回 True。
    """
    bbox = np.sort(mesh.bounding_box.extents)
    min1, min2, max_edge = bbox[0], bbox[1], bbox[2]

    # 防止除以零
    if min1 < eps or min2 < eps:
        return True

    if (max_edge / min1 > threshold) and (max_edge / min2 > threshold):
        return True
    return False


def merge_small_parts(parts, vol_thresh=0.005, contact_thresh=0.2, dist_thresh=1e-3):
    N = len(parts)
    mesh_vols = np.array([cat_approximate_mesh_volume(p) for p in parts])
    # colli_vols = np.array([cat_convex_hull_volume(p) for p in parts])
    edge_rates = np.array([edge_min_max_rate(p) for p in parts])
    aabb_vols = np.array([cat_aabb_mesh_volume(p) for p in parts])
    del_parts = np.array([filter_invalid_part(p) for p in parts])
    mesh_areas = np.array([p.area for p in parts])

    vol_ratios_aabb = aabb_vols / (mesh_vols + 1e-10)
    vol_ratios_all = mesh_vols / np.sum(mesh_vols)
    area_ratios_all = mesh_areas / np.sum(mesh_areas)

    centers = np.array([p.centroid for p in parts])
    tree = KDTree(centers)
    merge_results = []
    del_results = []
    for i, del_flag in enumerate(del_parts):
        # if i not in [22]:
        #     continue
        if del_flag:
            parts[i] = None
            del_results.append(i)
            continue
        if area_ratios_all[i] > 0.1:
            continue

        if vol_ratios_all[i] > 0.1:
            continue
        """
        area_ratios_all[i] < 0.0001
        vol_ratios_aabb[i] > 50
        vol_ratios_all[i] < 0.00008  正好对mv有效
        """
        if aabb_vols[i] < 1e-6 or mesh_areas[i] < 1e-6 or edge_rates[i] or vol_ratios_aabb[i] > 50 or (
                vol_ratios_all[i] < 0.0001 and area_ratios_all[i] < 0.0005):

            part = parts[i]

            # 找最近邻（排除自身）
            dists, idxs = tree.query(part.centroid, k=min(20, N))
            idxs = [j for j in idxs if j != i]

            best_j = None
            best_contact = 0.0

            # 采样小部件表面点
            sample_pts, _ = trimesh.sample.sample_surface(part, 200)

            for j in idxs:
                neighbor = parts[j]
                if neighbor is None:
                    continue
                # 包围盒距离过滤
                bbox_i = part.bounding_box.bounds
                bbox_j = neighbor.bounding_box.bounds
                gap = np.maximum(0, np.maximum(bbox_j[0] - bbox_i[1], bbox_i[0] - bbox_j[1]))
                if np.linalg.norm(gap) > 5e-2:  # 5e-3 最近-> 2e-2
                    continue
                # 计算点到邻居表面的距离
                points_on_surface, dist, face_ids = neighbor.nearest.on_surface(sample_pts)
                contact_ratio = np.mean(dist < dist_thresh)

                if contact_ratio > best_contact:
                    best_contact = contact_ratio
                    best_j = j

            if best_j is not None:  # and best_contact > contact_thresh:
                print(f"Merge part {i} -> {best_j}, contact={best_contact:.5f}")
                # parts[best_j] = parts[best_j].union(part)
                merge_results.append([best_j, i])
                parts[i] = None  # 删除当前部件

    return merge_results, del_results


def geom_to_trimesh(vertices, faces):
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return new_mesh


def save_face2cls(merged_parts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    face2cls = {}
    for cls, part in enumerate(merged_parts):
        for face in part:
            if face not in face2cls:
                face2cls[str(face)] = cls
    json.dump(face2cls, open(f"{save_dir}/final_post_face2cls.json", "w"), indent=4)


def post_process_main(
    merged_parts, mesh_path, save_dir, use_relative_threshold=False
):
    mesh = read_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces
    part_meshes = []
    for part_faces in merged_parts:
        sub_faces = faces[part_faces]

        # 顶点子集（注意要重新映射索引）
        unique_vids, inv_idx = np.unique(sub_faces.flatten(), return_inverse=True)
        sub_vertices = vertices[unique_vids]
        sub_faces = inv_idx.reshape(sub_faces.shape)
        sub_mesh = trimesh.Trimesh(
            vertices=sub_vertices,
            faces=sub_faces,
            process=False
        )
        # out_path = os.path.join(save_dir, "objs2", f"{ind}.obj")
        # sub_mesh.export(out_path)
        part_meshes.append(sub_mesh)
    dist_thresh = 1e-3
    if use_relative_threshold:
        dist_thresh *= float(np.max(mesh.bounding_box.extents))
    merge_results, del_results = merge_small_parts(
        deepcopy(part_meshes), dist_thresh=dist_thresh
    )
    for res in merge_results:
        tmp = []
        tmp.extend(merged_parts[res[0]])
        tmp.extend(merged_parts[res[1]])
        # part_meshes[res[0]].show()
        # part_meshes[res[1]].show()
        tmp = set(tmp)
        merged_parts[res[0]] = list(tmp)
        merged_parts[res[1]] = []
    for del_ind in del_results:
        merged_parts[del_ind] = []
    new_merge_parts = []
    for part in merged_parts:
        if len(part) > 0:
            new_merge_parts.append(part)
    save_face2cls(new_merge_parts, save_dir)
    return new_merge_parts
