from collections import defaultdict
from tqdm import tqdm
import numpy as np
import OpenEXR
import Imath
import os
import json
import argparse

from .mesh_handle import read_mesh, get_face_areas_list, build_mesh_from_part
from .union_find import UnionFind


def read_exr(exr_path):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
    return r


def compute_iou_face_type(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union, inter / min(len(set_a), len(set_b))


def compute_iou_area_type(set_a, set_b, areas):
    inter = set_a & set_b
    union = set_a | set_b
    if len(inter) == 0 or len(union) == 0:
        return 0.0, 0.0
    inds = np.array(list(inter)) - 1
    inter_area = np.sum(areas[inds])
    inds = np.array(list(union)) - 1
    union_area = np.sum(areas[inds])
    inds_a = np.array(list(set_a)) - 1
    inds_b = np.array(list(set_b)) - 1
    a_area = np.sum(areas[inds_a])
    b_area = np.sum(areas[inds_b])
    if union_area == 0:
        return 0.0, 0.0
    return inter_area / union_area, inter_area / min(a_area, b_area)


def merge_masks(args, mask_faces_list, iou_thresh=0.5):
    """
    mask_faces_list: List[Set[int]]
        每个元素是一个 mask 对应的 3D face index 集合
    iou_thresh: float
        IoU 阈值，用于判断两个 mask 是否属于同一个部件
    return:
        merged_groups: List[List[int]]  # 每个组包含原始 mask 的索引
    """
    n = len(mask_faces_list)
    uf = UnionFind(n)
    areas = []
    if args.cmp_iou_type == "area_type":
        areas = get_face_areas_list(args.mesh_path)

    # 两两比较
    for i in range(n):
        for j in range(i + 1, n):
            if args.cmp_iou_type == "area_type":
                iou, part_iou = compute_iou_area_type(mask_faces_list[i], mask_faces_list[j], areas)
            else:
                iou, part_iou = compute_iou_face_type(mask_faces_list[i], mask_faces_list[j])

            if iou >= iou_thresh or part_iou > 0.9:
                uf.union(i, j)

    # 聚合到连通分量
    groups = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(i)

    return list(groups.values())


def get_mask_faces(mask_path, exr_path, mask_faces_list, area_merge=True):
    cur_sam_masks = []
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
        mask_faces_list.append(set(mask2face[key]))
        cur_sam_masks.append(set(np.array(mask2face[key]) - 1))
    return cur_sam_masks


def save_face2cls(mask_faces_list, save_dir, iou_type):
    os.makedirs(save_dir, exist_ok=True)
    face2cls = {}
    cls = 0

    for ind in range(len(mask_faces_list)):
        for face in mask_faces_list[ind]:
            face = int(face) - 1
            if face not in face2cls:
                face2cls[face] = cls
        cls += 1
    json.dump(face2cls, open(f"{save_dir}/sam_face2cls_{iou_type}.json", "w"), indent=4)


def filter_group_by_majority(merged_groups, mask_faces_list, threshold=0.2):
    """
    对每个 merged group（list of indices into mask_faces_list）进行 face-level多数投票过滤。
    参数：
      - merged_groups: List[List[int]]  每个 group 是若干 mask 索引的列表（来自 mask_faces_list）
      - mask_faces_list: List[Set[int]]  原始每个 mask 对应的 face 集（face id 为 int）
      - threshold: float  多数阈值（默认 0.5，表示 >50% 的 mask 要包含该 face 才保留）
    返回：
      - filtered_groups: List[Set[int]] 每个 group 过滤后的 face 集合
    注意：
      - face id 在你的代码里似乎是从1开始的（exr里），需与 areas 索引保持一致（areas[face_id-1])
    """
    filtered_groups = []

    for group in merged_groups:
        # group 是若干 mask 的索引
        # 统计该 group 内每个 face 被多少 mask 覆盖（或被多少 mask 的面积票数支持）
        face_counter = defaultdict(float)  # face_id -> vote (float if area-weighted)
        total_masks = len(group)
        if total_masks == 0:
            continue

        # 累计票数
        for mask_idx in group:
            faces = mask_faces_list[mask_idx]
            # 普通计数：每个 mask 为它覆盖的 face +1
            for f in faces:
                face_counter[f] += 1.0

        # 计算阈值：如果不加权，阈值为 ceil(threshold * total_masks)
        need_votes = 1  # max(1, threshold * total_masks)
        kept_faces = {f for f, v in face_counter.items() if v > need_votes}
        if len(kept_faces) > 0:
            filtered_groups.append(kept_faces)

    return filtered_groups


def sam_3d_mask_main(args):
    mesh_path = args.mesh_path
    exr_dir = args.exr_dir
    seg_dir = args.seg_dir

    mask_faces_list = []
    multi_view_masks = []
    for i in tqdm(range(args.render_num)):
        exr_path = os.path.join(exr_dir, f"view_{i}_faceID0001.exr")
        mask_path = os.path.join(seg_dir, f'view_{i}_s.npy')
        cur_view_masks = get_mask_faces(mask_path, exr_path, mask_faces_list, area_merge=True)
        multi_view_masks.append(cur_view_masks)

    if args.show_part:
        # for test
        mesh = read_mesh(mesh_path)
        for cur_faces in mask_faces_list:
            part_mesh = build_mesh_from_part(mesh, cur_faces, sam_part=True)
            part_mesh.show()

    best_cls_num = len(mask_faces_list)
    merged = None
    for i in range(args.max_merge_num):
        merged = merge_masks(args, mask_faces_list, iou_thresh=0.2)
        tmp_cls_num = len(merged)
        if tmp_cls_num < best_cls_num:
            print(best_cls_num, tmp_cls_num, f"For the {i}-th time.")
            tmp_mask_faces_list = []
            for group in merged:
                tmp = []
                for ind in group:
                    tmp.extend(mask_faces_list[ind])
                tmp_mask_faces_list.append(set(tmp))
            mask_faces_list = tmp_mask_faces_list
            best_cls_num = tmp_cls_num
    save_face2cls(mask_faces_list, args.save_dir, args.cmp_iou_type)
    return mask_faces_list, multi_view_masks
