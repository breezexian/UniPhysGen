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
    json.dump(face2cls, open(os.path.join(save_dir, "part_refine_face2cls.json"), "w"), indent=4)


def refine_parts_with_connectivity(vertices, faces, face_labels, save_dir, eps=5e-3):
    """
    后处理 PartField 的分割结果，保证每个部件几何上真正连通。

    Args:
        vertices: (N, 3) float array
        faces: (M, 3) int array
        face_labels: (M,) int array, PartField 输出的 face 分类
        eps: float, 容差阈值（用于修补缝隙）

    Returns:
        parts: list of trimesh.Trimesh, 每个子部件
    """
    parts = []
    for label in np.unique(face_labels):

        # 当前部件的所有 face 索引
        face_idx = np.where(face_labels == label)[0]

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


            # comp_faces = sub_faces[comp_faces_mask]
            # if len(comp_faces) == 0:
            #     continue
            #
            # comp_unique_vids, inv_idx = np.unique(comp_faces.flatten(), return_inverse=True)
            # comp_vertices = sub_vertices[comp_unique_vids]
            # comp_faces = inv_idx.reshape(comp_faces.shape)
            # random_color = np.random.rand(3)
            # submesh = trimesh.Trimesh(vertices=comp_vertices,
            #                           faces=comp_faces,
            #                           process=True)
            # submesh.visual.vertex_colors = random_color
            # submesh.show()
            # parts.append(submesh)

    # face2cls = {}
    # for ind in range(len(parts)):
    #     for face in parts[ind]:
    #         face2cls[face] = ind
    # face2cls = sorted(face2cls.items(), key=lambda x: x[0])
    # cls_list = []
    # for item in face2cls:
    #     cls_list.append(item[1])
    # cls_list = np.array(cls_list)
    # np.save("partrefine.npy", cls_list)
    # json.dump(face2cls, open(os.path.join(save_dir, "part_refine_face2cls.json"), "w"), indent=4)
    save_face2cls(parts, save_dir)
    return parts


def part_refine_main(mesh_path, best_cluster_file, save_dir):
    mesh = read_mesh(mesh_path)
    faces = mesh.faces
    vertices = mesh.vertices
    face_labels = np.load(best_cluster_file)
    new_parts = refine_parts_with_connectivity(vertices, faces, face_labels, save_dir)
    return new_parts

