import numpy as np
import json
import os
import trimesh
from scipy.spatial import cKDTree
from collections import defaultdict, deque
from .mesh_handle import read_mesh


def save_face2cls(merged_parts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    face2cls = {}
    for cls, part in enumerate(merged_parts):
        for face in part:
            if face not in face2cls:
                face2cls[str(face)] = cls
    json.dump(face2cls, open(os.path.join(save_dir, "part_refine_face2cls_after_merge.json"), "w"), indent=4)


def refine_parts_with_connectivity(vertices, faces, mask2face, save_dir, eps=1e-2):

    parts = []
    for label in mask2face.keys():

        # 当前部件的所有 face 索引
        face_idx = mask2face[label]

        if len(face_idx) == 0:
            continue
        sub_faces = faces[face_idx]

        # 顶点子集（注意要重新映射索引）
        unique_vids, inv_idx = np.unique(sub_faces.flatten(), return_inverse=True)
        sub_vertices = vertices[unique_vids]
        sub_faces = inv_idx.reshape(sub_faces.shape)

        # 建立 sub_face 到 global_face 的映射
        face_map = {i: orig for i, orig in enumerate(face_idx)}

        # -------- Step1: 基于拓扑的邻接 --------
        adjacency = defaultdict(set)
        for i, f in enumerate(sub_faces):
            for j in range(3):
                v1, v2 = f[j], f[(j + 1) % 3]
                # 共享边 → 邻居
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)

        # -------- Step2: 基于距离的邻接 --------
        tree = cKDTree(sub_vertices)
        pairs = tree.query_pairs(r=eps)  # 所有距离小于 eps 的点对
        for v1, v2 in pairs:
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)

        # -------- Step3: DFS 找连通分量 --------
        visited = np.zeros(len(sub_vertices), dtype=bool)
        components = []
        for vid in range(len(sub_vertices)):
            if not visited[vid]:
                comp = []
                q = deque([vid])
                visited[vid] = True
                while q:
                    cur = q.popleft()
                    comp.append(cur)
                    for nb in adjacency[cur]:
                        if not visited[nb]:
                            visited[nb] = True
                            q.append(nb)
                components.append(comp)
        if len(components) > 5:
            parts.append(face_idx)
            continue
        # -------- Step4: 根据顶点集合切分 faces --------
        for comp in components:
            comp_set = set(comp)
            comp_faces_mask = np.all(np.isin(sub_faces, list(comp_set)), axis=1)
            comp_faces_indx = []
            for ind, flag in enumerate(comp_faces_mask):
                if flag:
                    comp_faces_indx.append(face_map[ind])
            if len(comp_faces_indx) == 0:
                continue
            parts.append(comp_faces_indx)

    save_face2cls(parts, save_dir)
    return parts


def part_refine_after_merge_main(
    mesh_path, face2cls_file, save_dir, use_relative_threshold=False
):
    with open(face2cls_file) as f:
        mask2face = {}
        face2cls = json.load(f)
        for face in face2cls:
            cls = face2cls[face]
            if cls not in mask2face:
                mask2face[cls] = []
            mask2face[cls].append(int(face))
    mesh = read_mesh(mesh_path)
    faces = mesh.faces
    vertices = mesh.vertices
    eps = 1e-2
    if use_relative_threshold:
        eps *= float(np.max(mesh.bounding_box.extents))
    new_parts = refine_parts_with_connectivity(
        vertices, faces, mask2face, save_dir, eps=eps
    )
    return new_parts
