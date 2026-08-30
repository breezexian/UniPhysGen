import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import trimesh
from collections import defaultdict

from .mesh_handle import read_mesh, get_face_areas_list, build_mesh_from_part


def compute_iou(set_a, set_b, areas):
    inter = set_a & set_b
    union = set_a | set_b
    if len(inter) == 0 or len(union) == 0:
        return 0.0
    inds = np.array(list(inter))
    inter_area = np.sum(areas[inds])
    inds = np.array(list(union))
    union_area = np.sum(areas[inds])
    inds = np.array(list(set_a))
    ori_area = np.sum(areas[inds])
    if union_area == 0:
        return 0.0
    return inter_area / ori_area


def compute_iou_deprecated(set_a, set_b, areas):
    inter = set_a & set_b
    union = set_a | set_b
    if len(inter) == 0 or len(union) == 0:
        return 0.0
    inds = np.array(list(inter))
    inter_area = np.sum(areas[inds])
    inds = np.array(list(union))
    union_area = np.sum(areas[inds])
    inds_a = np.array(list(set_a))
    inds_b = np.array(list(set_b))
    a_area = np.sum(areas[inds_a])
    b_area = np.sum(areas[inds_b])
    if union_area == 0:
        return 0.0
    return inter_area / min(a_area, b_area)


def merge_masks(mesh, sam_masks, part_masks, part_cls, areas, iou_thresh=0.5):
    n = len(part_masks)
    m = len(sam_masks)

    # 两两比较
    merged_groups = {}
    for i in range(n):
        for j in range(m):
            iou = compute_iou(part_masks[i], sam_masks[j], areas)
            if iou >= iou_thresh:
                if j not in merged_groups:
                    merged_groups[j] = []
                merged_groups[j].append(part_cls[i])

                # part_mesh = build_mesh_from_part(mesh, part_masks[i])
                # sam_mesh = build_mesh_from_part(mesh, sam_masks[j])
                #
                # part_mesh.show()
                # sam_mesh.show()
                break

    return merged_groups


def merge_masks_deprecated(mesh, sam_masks, part_masks, part_cls, areas, iou_thresh=0.5):
    n = len(part_masks)
    m = len(sam_masks)

    # 两两比较
    sam_to_parts = {}
    for i in range(n):
        for j in range(m):
            iou = compute_iou(part_masks[i], sam_masks[j], areas)
            if iou >= iou_thresh:
                if j not in sam_to_parts:
                    sam_to_parts[j] = []
                sam_to_parts[j].append(part_cls[i])

    # step2: 构建 sam 之间的连通关系（因为共享 part）
    adj = defaultdict(set)
    for j in sam_to_parts.keys():
        adj[j] = set()  # 先初始化，保证孤立点也存在

    for j, parts in sam_to_parts.items():
        for other_j, other_parts in sam_to_parts.items():
            if j < other_j and set(parts) & set(other_parts):  # 如果共享 part
                adj[j].add(other_j)
                adj[other_j].add(j)
    # step3: 找连通分量（最终合并结果）
    visited = set()
    merged_groups = {}
    for j in range(m):
        if j in visited or j not in sam_to_parts:
            continue
        stack = [j]
        group_sams = set()
        group_parts = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group_sams.add(cur)
            group_parts.update(sam_to_parts[cur])
            for nb in adj[cur]:
                if nb not in visited:
                    stack.append(nb)
        # merged_groups.append({
        #     "sams": list(group_sams),
        #     "parts": list(group_parts)
        # })
        cls_str = "-".join([str(i) for i in group_sams])
        merged_groups[cls_str] = (list(group_parts))

    return merged_groups


def save_face2cls(merged_parts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    face2cls = {}
    for cls, part in enumerate(merged_parts):
        for face in part:
            if face not in face2cls:
                face2cls[str(face)] = cls
    json.dump(face2cls, open(f"{save_dir}/final_face2cls.json", "w"), indent=4)


def merge_parts_main(mesh_path, sam_parts, partfield_parts, save_dir):
    mesh = read_mesh(mesh_path)
    areas = get_face_areas_list(mesh_path)
    sam_faces = []
    for i in range(len(sam_parts)):
        sam_faces.extend(sam_parts[i])
    sam_faces = set(sam_faces)

    partfield_parts_sam = []
    part_cls = []
    for cls, part in enumerate(partfield_parts):
        tmp_part = []
        for face in part:
            if face in sam_faces:
                tmp_part.append(face)
        if len(tmp_part) > 0:
            partfield_parts_sam.append(set(tmp_part))
            part_cls.append(cls)
    merge_groups = merge_masks(mesh, sam_parts, partfield_parts_sam, part_cls, areas, iou_thresh=0.5)
    merge_parts = []
    remain_cls = set()
    for group in merge_groups:
        tmp_part = []
        for cls in merge_groups[group]:
            tmp_part.extend(partfield_parts[cls])
            remain_cls.add(cls)
        merge_parts.append(tmp_part)
    out_cls = set(list(range(len(partfield_parts)))) - remain_cls
    for cls in out_cls:
        merge_parts.append(partfield_parts[cls])
    save_face2cls(merge_parts, save_dir)
    return merge_parts

