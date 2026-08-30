from collections import defaultdict
from tqdm import tqdm
import numpy as np
import OpenEXR
import Imath
import os
import json
import argparse

from .mesh_handle import read_mesh, get_face_areas_list, build_mesh_from_part


def read_exr(exr_path):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
    return r


def get_mask_faces(mask_path, exr_path, area_merge=True):
    sam_masks = []
    mask = np.load(mask_path)
    exr = read_exr(exr_path)
    size = np.shape(mask)
    face2mask = {}
    mask2area = {}
    for i in range(size[0]):
        for j in range(size[1]):
            if exr[i][j] == 0 or mask[i][j] == -1:
                continue
            cls = mask[i][j]
            face_id = int(exr[i][j])

            # get face2mask
            if face_id not in face2mask:
                face2mask[face_id] = {}
            if cls not in face2mask[face_id]:
                face2mask[face_id][cls] = 1
            else:
                face2mask[face_id][cls] += 1

            # get mask2area 像素点的个数
            if cls not in mask2area:
                mask2area[cls] = {}
            if face_id not in mask2area[cls]:
                mask2area[cls][face_id] = 1
            else:
                mask2area[cls][face_id] += 1

    mask2face = {}
    face2mask_new = {}
    for face_id in face2mask:
        cur = face2mask[face_id]
        cur = sorted(cur.items(), key=lambda item: item[1], reverse=True)
        cls = cur[0][0]
        # px_num = cur[0][1]
        # if px_num < 2:
        #     continue
        face2mask_new[face_id] = cls
        if cls not in mask2face:
            mask2face[cls] = []
        mask2face[cls].append(face_id)
    if area_merge:
        for cls in mask2area:
            tot_area = np.sum(list(mask2area[cls].values()))
            out_area = 0
            newcls2area = {}
            remain_faces = []
            # 生成mask2face投票的时候，有些mask会消失
            if cls not in mask2face:
                continue
            for face_id in mask2area[cls]:
                if face_id not in face2mask_new:
                    continue
                if face_id not in mask2face[cls]:
                    cur_area = mask2area[cls][face_id]
                    out_area += cur_area
                    new_cls = face2mask_new[face_id]
                    if new_cls not in newcls2area:
                        newcls2area[new_cls] = cur_area
                    else:
                        newcls2area[new_cls] += cur_area
                else:
                    remain_faces.append(face_id)
            area_rate = out_area / tot_area

            if area_rate > 0.5:
                new_cls = sorted(newcls2area.items(), key=lambda item: item[1], reverse=True)

                new_cls = new_cls[0][0]
                for face_id in remain_faces:
                    mask2face[new_cls].append(face_id)
                    face2mask_new[face_id] = new_cls
                del mask2face[cls]
    for key in mask2face:
        sam_masks.append(set(np.array(mask2face[key]) - 1))
    return sam_masks


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


def compute_iou_merge_apriori(set_a, set_b, areas):
    inter = set_a & set_b
    union = set_a | set_b
    if len(inter) == 0 or len(union) == 0:
        return 0.0, 0.0
    inds = np.array(list(inter))
    inter_area = np.sum(areas[inds])
    inds = np.array(list(union))
    union_area = np.sum(areas[inds])
    inds_a = np.array(list(set_a)) - 1
    inds_b = np.array(list(set_b)) - 1
    a_area = np.sum(areas[inds_a])
    b_area = np.sum(areas[inds_b])
    if union_area == 0:
        return 0.0, 0.0
    return inter_area / union_area, inter_area / min(a_area, b_area)


def cpm_iou_for_part_ids(set1, set2):
    """计算两个集合的 IoU"""
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0


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
                break
    return merged_groups


def save_face2cls(merged_parts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    face2cls = {}
    for cls, part in enumerate(merged_parts):
        for face in part:
            if face not in face2cls:
                face2cls[str(face)] = cls
    json.dump(face2cls, open(f"{save_dir}/final_face2cls.json", "w"), indent=4)


def get_part_areas(partfield_parts, areas):
    part_areas = []
    for part in partfield_parts:
        inds = np.array(list(part))
        area = np.sum(areas[inds])
        part_areas.append(area)
    return np.array(part_areas)


def sort_itemsets_by_support(L, support_data, reverse=True):
    """
    对每一层频繁项集按支持度排序
    :param L: list[list[frozenset]]，每个子列表是一种长度的频繁项集
    :param support_data: dict，项集->支持度
    :param reverse: bool，是否从高到低排序
    :return: list[list[frozenset]]，排序后的结构
    """
    sorted_L = []
    for itemsets in L:
        if len(itemsets) == 0:
            continue
        sorted_itemsets = sorted(
            itemsets,
            key=lambda x: support_data.get(x, 0),
            reverse=reverse
        )
        sorted_L.append(sorted_itemsets)
    return sorted_L


def apriori_merge_parts_main(args, multi_view_masks, partfield_parts, save_dir):
    mesh_path = args.mesh_path
    exr_dir = args.exr_dir
    seg_dir = args.seg_dir

    mesh = read_mesh(mesh_path)
    areas = get_face_areas_list(mesh_path)
    part_areas = get_part_areas(partfield_parts, areas)
    combined_list = []
    for i in tqdm(range(args.render_num)):
        exr_path = os.path.join(exr_dir, f"view_{i}_faceID0001.exr")
        mask_path = os.path.join(seg_dir, f'view_{i}_s.npy')
        sam_parts = multi_view_masks[i]  # get_mask_faces(mask_path, exr_path, area_merge=True)
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
        for key in merge_groups.keys():
            combined_list.append(merge_groups[key])
    from .apriori_match2 import apriori_mlxtend
    frequent_combined_list, support_data = apriori_mlxtend(combined_list, min_support=3)
    frequent_combined_list = sort_itemsets_by_support(frequent_combined_list, support_data)
    freq_num = len(frequent_combined_list)

    # 剔除子集
    filtered = []
    for i in range(freq_num - 1, -1, -1):
        for itemset in frequent_combined_list[i]:
            if not any(itemset.issubset(t) for t in filtered):  # s 是 t 的真子集则丢弃
                filtered.append(itemset)

    # 合并iou高的集合
    final_groups = []
    used = set()
    for i, s1 in enumerate(filtered):
        if i in used:
            continue
        merged_set = set(s1)
        for j, s2 in enumerate(filtered):
            if j <= i or j in used:
                continue
            iou, inner_iou = compute_iou_merge_apriori(s1, s2, part_areas)
            inter = s1 & s2  # 集合交集
            if iou > 0.2 or inner_iou > 0.5:
                merged_set |= s2
                used.add(j)
            elif len(inter) > 0:
                s1_area = np.sum(part_areas[list(s1)])
                s2_area = np.sum(part_areas[list(s2)])
                if s1_area > s2_area:
                    filtered[j] = filtered[j] - inter
                else:
                    merged_set = merged_set - inter

        final_groups.append(frozenset(merged_set))
        used.add(i)

    # covered_parts = set()
    # for i in range(freq_num - 1, -1, -1):
    #     for itemset in frequent_combined_list[i]:
    #         if not any(p in covered_parts for p in itemset):
    #             final_groups.append(itemset)
    #             covered_parts.update(itemset)

    merge_parts = []
    remain_cls = set()
    for group in final_groups:
        tmp_part = []
        for cls in group:
            tmp_part.extend(partfield_parts[cls])
            remain_cls.add(cls)
        merge_parts.append(tmp_part)
    out_cls = set(list(range(len(partfield_parts)))) - remain_cls
    for cls in out_cls:
        merge_parts.append(partfield_parts[cls])
    save_face2cls(merge_parts, save_dir)
    return merge_parts
