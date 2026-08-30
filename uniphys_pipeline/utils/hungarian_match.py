import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import trimesh
import argparse
from tqdm import tqdm

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
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def sam2_partfield_loss_deprecated(sam_masks, partfield_masks, areas, sam_weights, part_weights):
    match_dict = {}
    m, n = len(sam_masks), len(partfield_masks)
    # IoU 矩阵
    cost_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            iou = compute_iou(sam_masks[i], partfield_masks[j], areas)
            # * min(sam_weights[i], part_weights[j])/max(sam_weights[i], part_weights[j])  # 因为 Hungarian 算法是最小化
            cost_matrix[i, j] = 1 - iou

    # 匹配
    matched_idx = np.argmin(cost_matrix, axis=1)  # 每个 sam 对应的最佳 partfield 索引
    matched_iou = 1 - cost_matrix[np.arange(m), matched_idx]
    # matched_iou_weights = cls_weights * matched_iou
    # L_iou = 1 - np.mean(matched_iou_weights)  # 越小越好
    L_iou = 1 - np.mean(matched_iou)
    for cls in range(len(sam_masks)):
        match_dict[cls] = {"sam": list(sam_masks[cls]), "part": list(partfield_masks[matched_idx[cls]])}
    json.dump(match_dict, open("match.json", "w"), indent=4)
    return L_iou, matched_iou


def refine_sam_masks(sam_masks, areas, surface_areas):
    sam_masks_areas = []
    for mask in sam_masks:
        mask = list(mask)
        sam_masks_areas.append(np.sum(areas[mask]))
    inds = np.argsort(sam_masks_areas)[::-1]
    new_sam_mask = []
    tot_area = 0
    for ind in inds:
        if tot_area / surface_areas > 0.99:
            break
        tot_area += sam_masks_areas[ind]
        new_sam_mask.append(sam_masks[ind])
    return new_sam_mask


def sam2_partfield_loss(sam_masks, partfield_masks, areas, surface_areas):
    m, n = len(sam_masks), len(partfield_masks)
    # IoU 矩阵
    cost_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            iou = compute_iou(sam_masks[i], partfield_masks[j], areas)
            cost_matrix[i, j] = 1 - iou  # 因为 Hungarian 算法是最小化

    # Hungarian 匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    remain_faces = []
    for ind in row_ind:
        remain_faces.extend(sam_masks[ind])
    remain_faces = list(set(remain_faces))
    remain_inds = np.array(remain_faces)
    remain_areas = np.sum(areas[remain_inds])
    alpha = remain_areas / surface_areas  # * len(row_ind)/len(sam_masks) * 0.5
    # IoU Loss
    # print(alpha)
    matched_ious = [1 - cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    L_iou = 1 - np.mean(matched_ious) * alpha
    return L_iou, matched_ious


def sam2_partfield_loss_multi_view(multi_view_masks, partfield_parts, areas, surface_areas):
    tot_loss = 0
    for i in range(len(multi_view_masks)):
        sam_parts = multi_view_masks[i]
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
        L_iou, matched_cost = sam2_partfield_loss(sam_parts, partfield_parts_sam, areas, surface_areas)
        tot_loss += L_iou
    return tot_loss / len(multi_view_masks)


def hm_main(mesh_path, sam_part_file, partfield_part_dir, entity, init_cluseter_num, cluster_num_inc_range,
            multi_view_masks=None,
            refine=True):
    areas = get_face_areas_list(mesh_path)
    sam_masks = []
    left = cluster_num_inc_range[0]
    right = cluster_num_inc_range[1]
    with open(sam_part_file) as f:
        mask2face = {}
        face2cls = json.load(f)
        sam_faces = set(list(map(lambda x: int(x), sorted(face2cls.keys()))))
        for face in face2cls:
            cls = face2cls[face]
            if cls not in mask2face:
                mask2face[cls] = []
            mask2face[cls].append(int(face))
        for cls in mask2face:
            sam_masks.append(set(mask2face[cls]))
    surface_areas = np.sum(areas[list(sam_faces)])
    if refine:
        sam_masks = refine_sam_masks(sam_masks, areas, surface_areas)
        print(f"调整后sam masks长度：{len(sam_masks)}")
        init_cluseter_num = len(sam_masks)

    hm_loss = np.inf
    best_cluster_file = None
    for cluster_num in range(max(1, init_cluseter_num - left), init_cluseter_num + right):
        pth = os.path.join(partfield_part_dir, f"{entity}_0_{str(cluster_num).zfill(2)}.npy")
        masks = np.load(pth)
        mask2face = {}
        partfield_masks = []
        for ind, cls in enumerate(masks):
            face = ind
            if face in sam_faces:
                if cls not in mask2face:
                    mask2face[cls] = []
                mask2face[cls].append(face)
        for cls in mask2face:
            partfield_masks.append(set(mask2face[cls]))
        if multi_view_masks is None:
            L_iou, matched_cost = sam2_partfield_loss(sam_masks, partfield_masks, areas, surface_areas)
        else:
            L_iou = sam2_partfield_loss_multi_view(multi_view_masks, partfield_masks, areas, surface_areas)
        print(pth, "L_iou:", L_iou)

        if hm_loss > L_iou:
            hm_loss = L_iou
            best_cluster_file = pth

    return best_cluster_file, hm_loss