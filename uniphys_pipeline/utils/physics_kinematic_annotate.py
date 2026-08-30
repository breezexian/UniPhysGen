import open3d as o3d
import numpy as np
import os
from scipy.spatial import ConvexHull
import json
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import trimesh
import shutil
import logging
import argparse
import trimesh
import re


def fit_planes(pcd, distance_threshold=0.01, max_planes=5, min_inliers=100):
    """
    Use RANSAC hierarchical segmentation to fit multiple planes
    Parameters:
    pcd (open3d.geometry.PointCloud): input point cloud
    distance_threshold (float): RANSAC inlier distance threshold
    max_planes (int): maximum number of planes
    min_inliers (int): minimum inlier threshold for a plane
    Returns:
    planes (list): list of plane parameters, each plane is in the format of (A, B, C, D)
    """
    planes = []
    points = []
    remaining_pcd = pcd

    for _ in range(max_planes):
        if len(remaining_pcd.points) < 3 * min_inliers:  # Stop when the remaining points are insufficient
            break

        # Split the current largest plane
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )

        if len(inliers) < min_inliers:
            break

        planes.append(plane_model)
        points.append(inliers)

        # remaining_pcd = remaining_pcd.select_down_sample(inliers, invert=True)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    return planes, points


def rotate_vector(v, k, theta):
    """
    Calculate the new vector after the vector v is rotated around the axis k by the angle theta
    :param v: vector to be rotated (1x3 numpy array)
    :param k: unit vector of the rotation axis (1x3 numpy array)
    :param theta: rotation angle (radians)
    :return: rotated vector (1x3 numpy array)
    """

    k = k / np.linalg.norm(k)

    v_rot = (v * np.cos(theta) +
             np.cross(k, v) * np.sin(theta) +
             k * np.dot(k, v) * (1 - np.cos(theta)))

    return v_rot


def create_3d_arrow_mesh(position, direction, color, target_bbox=None, arrow_length=None, stem_radius=None,
                         head_radius=None, head_length_ratio=0.1):
    """
    Create a 3D arrow with size adapting to a entity bounding box.
    :param position: Arrow base coordinates
    :param direction: Arrow direction vector (normalized)
    :param color: Arrow color (RGB)
    :param target_bbox: np.array shape (2, 3), [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    :param arrow_length: Total arrow length (if None, use target_bbox size)
    :param stem_radius: Cylinder radius (if None, use arrow_length * 0.03)
    :param head_radius: Cone bottom radius (if None, use arrow_length * 0.06)
    :param head_length_ratio: Ratio of head length to total length
    :return: trimesh.Trimesh arrow, rotation matrix
    """

    direction = np.array(direction, dtype=np.float64)
    if np.allclose(direction, 0):
        raise ValueError("The direction vector cannot be zero!")
    direction /= np.linalg.norm(direction)

    # --- 自动计算箭头尺寸 ---
    if target_bbox is not None:
        diag_len = np.linalg.norm(target_bbox[1] - target_bbox[0])
        if arrow_length is None:
            arrow_length = diag_len * 1.6  # 箭头长度约为部件尺寸一半
        if stem_radius is None:
            stem_radius = arrow_length * 0.006
        if head_radius is None:
            head_radius = arrow_length * 0.012
    else:
        if arrow_length is None:
            arrow_length = 1.0
        if stem_radius is None:
            stem_radius = 0.02
        if head_radius is None:
            head_radius = 0.05

    head_length = arrow_length * head_length_ratio
    stem_length = arrow_length - head_length

    # --- Create stem and head ---
    stem = trimesh.creation.cylinder(radius=stem_radius, height=stem_length, sections=32)
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=32)
    head.apply_translation([0, 0, stem_length / 2])

    arrow = trimesh.util.concatenate([stem, head])

    # --- rotate to target direction ---
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    dot_product = np.clip(np.dot(z_axis, direction), -1.0, 1.0)
    angle = np.arccos(dot_product)
    axis = np.cross(z_axis, direction)
    if np.linalg.norm(axis) < 1e-10:
        rotation = trimesh.transformations.rotation_matrix(np.pi if dot_product < 0 else 0, [1, 0, 0])
    else:
        axis /= np.linalg.norm(axis)
        rotation = trimesh.transformations.rotation_matrix(angle, axis)

    arrow.apply_transform(rotation)
    arrow.apply_translation(position)
    arrow.visual.vertex_colors = color

    return arrow, rotation


def vector_angle_np(a, b):
    a = np.array(a)
    b = np.array(b)
    # Check Dimensions
    if a.shape != b.shape:
        raise ValueError("Vector dimensions are different")
    # Calculate dot product and modulus
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Handling zero vectors
    if norm_a == 0 or norm_b == 0:
        raise ValueError("There is a zero vector and the angle cannot be calculated")
    # Calculating Angles
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # limite the range
    degrees = np.degrees(np.arccos(cos_theta))
    return degrees


def get_child_parent_mesh(part_seg_dir, entity, child_label):
    child_label_file = os.path.join(part_seg_dir, entity, 'objs', str(child_label) + '.obj')
    child_mesh = trimesh.load(child_label_file, force='mesh')
    part_objs = [file for file in os.listdir(os.path.join(part_seg_dir, entity, 'objs')) if ".obj" in file]
    other_parts = list(set(part_objs) - set([str(child_label) + '.obj']))
    part_meshes = []
    for idx in range(len(other_parts)):
        tmp_mesh = trimesh.load(os.path.join(part_seg_dir, entity, 'objs', other_parts[idx]), force='mesh')
        part_meshes.append(tmp_mesh)
    other_mesh = trimesh.util.concatenate(part_meshes)

    return child_mesh, other_mesh


def get_contact_region_old(child_mesh, other_mesh, threshold=1e-2):
    value, ind = ((torch.Tensor(other_mesh.vertices)[None].repeat(len(child_mesh.vertices), 1, 1) - torch.Tensor(
        child_mesh.vertices)[:, None].repeat(1, len(other_mesh.vertices), 1)) ** 2).sum(-1).min(0)
    jointregion = torch.Tensor(other_mesh.vertices)[torch.where(value.sqrt() < threshold)]
    count = 0
    while len(jointregion) <= 6:
        jointregion = torch.Tensor(other_mesh.vertices)[torch.where(value.sqrt() < threshold)]
        threshold += 1e-2
        count += 1
        if count > 1000:
            break
    return jointregion, count


def get_contact_region(child_mesh, other_mesh, threshold=1e-2, oversize=False,
                       child_sample=1000, other_sample=1000,
                       max_count=1000, device_id=0):
    """
    基于 mesh surface 采样，GPU 并行计算接触区域
    保留原顶点 + 采样额外点
    当接触点不足时，逐步增加阈值
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # child mesh 顶点 + 采样
    child_pts = np.array(child_mesh.vertices)
    if oversize:
        random_indices = np.random.choice(len(child_pts), size=len(child_pts)//2, replace=False)
        child_pts = child_pts[random_indices]
    if child_sample > 0:
        extra_child_pts = np.array(child_mesh.sample(child_sample))
        child_pts = np.vstack([child_pts, extra_child_pts])
    if oversize:
        child_pts = torch.tensor(child_pts, dtype=torch.float32)
    else:
        child_pts = torch.tensor(child_pts, dtype=torch.float32, device=device)

    # other mesh 顶点 + 采样
    other_pts = np.array(other_mesh.vertices)
    if oversize:
        random_indices = np.random.choice(len(other_pts), size=len(other_pts)//2, replace=False)
        other_pts = other_pts[random_indices]
    if other_sample > 0:
        extra_other_pts = np.array(other_mesh.sample(other_sample))
        other_pts = np.vstack([other_pts, extra_other_pts])
    if oversize:
        other_pts = torch.tensor(other_pts, dtype=torch.float32)
    else:
        other_pts = torch.tensor(other_pts, dtype=torch.float32, device=device)

    # 计算 pairwise 距离
    diff = child_pts[:, None, :] - other_pts[None, :, :]
    dists = torch.norm(diff, dim=-1)  # shape: [N_child, N_other]

    count = 0
    if oversize:
        jointregion = torch.zeros((0, 3))
    else:
        jointregion = torch.zeros((0, 3), device=device)

    while count < max_count:
        mask = dists.min(dim=0).values < threshold
        jointregion = other_pts[mask]
        if len(jointregion) > 6:  # 足够多的点
            break
        threshold += 1e-2
        count += 1
    if oversize:
        return jointregion, count
    return jointregion.cpu(), count


def get_plane_candidate_axes(child_mesh, other_mesh, threshold=1e-2):
    try:
        jointregion, count = get_contact_region(child_mesh, other_mesh, threshold)
    except Exception as e:
        print(e)
        # pattern1 = r'Tried to allocate\s+([\d\.]+)\s+GiB'
        # match1 = re.search(pattern1, str(e))
        # memory_gb = 0
        # if match1:
        #     memory_gb = float(match1.group(1))
        #     print(f"尝试分配的内存: {memory_gb} GB")
        jointregion, count = get_contact_region(child_mesh, other_mesh, threshold, oversize=True)

    print(f"循环次数：{count}")
    pcd = o3d.geometry.PointCloud()
    nn_points = jointregion.numpy()
    nn_points = nn_points.astype(np.float64)  # 强制转 float64 才可以进行o3d的操作
    pcd.points = o3d.utility.Vector3dVector(nn_points)
    planes, innerpoint = fit_planes(pcd, distance_threshold=0.05, max_planes=1,
                                    min_inliers=int(len(jointregion) * 0.1))

    para1, para2, para3, para4 = planes[0]
    plane_point = jointregion[innerpoint[0]].numpy().mean(0)
    inner = [para1, para2, para3] * plane_point
    para4 = -inner.sum()

    orilist = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    view = []
    angle = []
    for i in range(3):

        if vector_angle_np(planes[0][:3], orilist[i]) > 90:
            angle.append(180 - vector_angle_np(planes[0][:3], orilist[i]))
            view.append(-planes[0][:3])
        else:
            angle.append(vector_angle_np(planes[0][:3], orilist[i]))
            view.append(planes[0][:3])
    planeview = view[np.array(angle).argmin()]

    newplane_fine = [np.array([planeview[0], planeview[1], planeview[2], para4])]

    initview = np.cross(planeview, orilist[
        list(set([0, 1, 2]) - set([np.array(angle).argmin()]))[0]])
    if vector_angle_np(initview, orilist[
        list(set([0, 1, 2]) - set([np.array(angle).argmin()]))[1]]) > 90:
        initview = -initview

    # number of candidate axis
    num_candidate = 6
    for i in range(num_candidate):
        D = -(planeview * plane_point).sum()
        newplane_fine.append(np.concatenate(
            [rotate_vector(initview, planeview, i * np.pi / num_candidate),
             np.array([D])]))

    # child_mesh.visual.face_colors = color_list[0]
    newplane_fine.append(np.array([0, 0, 1, 0]))
    newplane_fine.append(np.array([1, 0, 0, 0]))
    newplane_fine.append(np.array([0, 1, 0, 0]))

    return newplane_fine, plane_point, jointregion


def get_candidate_axes_for_revolute(child_mesh, other_mesh, clusters=5, threshold=1e-2):
    newplane_fine, plane_point, jointregion = get_plane_candidate_axes(child_mesh, other_mesh, threshold)
    # if newplane_fine is None and plane_point is None:
    #     return None, None, None
    k = min(len(jointregion), clusters)  # number of groups
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(jointregion)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return newplane_fine, centers, plane_point


def clean_obj_id(obj_pth):
    with open(obj_pth, "r") as f:
        lines = f.readlines()
    cleaned = [l for l in lines if not l.startswith("o ")]
    with open(obj_pth, "w") as f:
        f.writelines(cleaned)


def judge_child_parent_contact(part_seg_dir, entity, child_label, parent_label):
    child_label_file = os.path.join(part_seg_dir, entity, 'objs', str(child_label) + '.obj')
    child_mesh = trimesh.load(child_label_file, force='mesh')
    parent_label_file = os.path.join(part_seg_dir, entity, 'objs', str(parent_label) + '.obj')
    parent_mesh = trimesh.load(parent_label_file, force='mesh')
    try:
        contact_region, count = get_contact_region(child_mesh, parent_mesh)
    except Exception as e:
        print(e)
        # pattern1 = r'Tried to allocate\s+([\d\.]+)\s+GiB'
        # match1 = re.search(pattern1, str(e))
        # memory_gb = 0
        # if match1:
        #     memory_gb = float(match1.group(1))
        #     print(f"尝试分配的内存: {memory_gb} GB")
        contact_region, count = get_contact_region(child_mesh, parent_mesh, oversize=True)
    if count <= 3:
        return child_mesh, parent_mesh, True
    return child_mesh, parent_mesh, False


def kinematic_annotation_main(part_seg_dir, save_res_dir, entity):
    os.makedirs(save_res_dir, exist_ok=True)
    kg_dir = os.path.join(save_res_dir, "kg_res", entity)
    json_file = os.path.join(kg_dir, "kinematic_info.json")
    kinematic_info = json.load(open(json_file, "r"))
    # gpt_basic_dir = os.path.join(save_res_dir, "gpt_basic_annotation")
    # rotate_dir = os.path.join(save_res_dir, 'obj_revolute_pivot')
    hinge_dir = os.path.join(save_res_dir, 'obj_hinge')
    prismatic_dir = os.path.join(save_res_dir, 'obj_prismatic')
    revolute_dir = os.path.join(save_res_dir, 'obj_revolute')

    color_list = [
        (1.0, 1.0, 0.0, 1.0),  # Yellow
        (1.0, 0.5, 0.0, 1.0),  # Orange
        (0.6, 0.2, 1.0, 1.0),  # Purple
        (0.0, 1.0, 1.0, 1.0),  # Cyan
        (1.0, 0.0, 1.0, 1.0),  # Magenta
        (0.275, 0.275, 0.275, 1.0),  # Dark Gray
        (0.4, 0.25, 0.1, 1.0),  # Brown
        (1.0, 0.0, 0.0, 1.0),  # Red
        (0.0, 1.0, 0.0, 1.0),  # Green
        (0.0, 0.0, 1.0, 1.0),  # Blue
    ]

    # jsonfile = os.path.join(gpt_basic_dir, f'{entity}.json')
    #
    # with open(jsonfile, 'r') as f:
    #     data = json.load(f)

    # annotated_parts = []
    for child_label in kinematic_info.keys():
        motion_info = kinematic_info[child_label]
        for mov_type in motion_info.keys():
            parent_label = motion_info[mov_type].get("parent", None)
            if parent_label is None:
                continue
            if mov_type == 'B':
                if os.path.exists(os.path.join(prismatic_dir, entity, str(child_label), 'axis.npy')):
                    continue
                img_dir = os.path.join(prismatic_dir, entity, str(child_label), 'imgs')
                if os.path.exists(img_dir):
                    shutil.rmtree(img_dir)
                os.makedirs(os.path.join(prismatic_dir, entity, str(child_label)), exist_ok=True)
                child_mesh, other_mesh, judge_flag = judge_child_parent_contact(part_seg_dir, entity, child_label,
                                                                                parent_label)
                # if child_mesh is None:
                #     continue
                print(judge_flag)
                if not judge_flag:
                    child_mesh, other_mesh = get_child_parent_mesh(part_seg_dir, entity, child_label)
                newplane_fine, plane_point, _ = get_plane_candidate_axes(child_mesh, other_mesh)
                # if newplane_fine is None and plane_point is None:
                #     continue
                for i in range(len(newplane_fine)):
                    arrow_mesh, rotation = create_3d_arrow_mesh(
                        position=plane_point + 0.04 * newplane_fine[0][:3] / np.linalg.norm(
                            newplane_fine[0][:3]), direction=newplane_fine[i][:3], color=color_list[i],
                        target_bbox=child_mesh.bounds)
                    arrow_mesh.export(
                        os.path.join(prismatic_dir, entity, str(child_label), f'axis_{i + 1}.obj'))

                child_mesh.export(os.path.join(prismatic_dir, entity, str(child_label), 'part.obj'))
                clean_obj_id(os.path.join(prismatic_dir, entity, str(child_label), 'part.obj'))
                # save axis
                np.save(os.path.join(prismatic_dir, entity, str(child_label), 'axis.npy'),
                        np.stack(newplane_fine)[:, :3])

                # np.save(os.path.join(prismatic_dir, entity, str(child_label), 'axis.npy'),
                #         np.concatenate([plane_point[None], np.stack(newplane_fine)[:, :3]], 0))

            if mov_type == 'C':

                axis_dir = os.path.join(revolute_dir, entity, str(child_label), 'axis')
                pivot_dir = os.path.join(revolute_dir, entity, str(child_label), 'pivot')
                if os.path.exists(os.path.join(axis_dir, 'axis.npy')) and \
                        os.path.exists(os.path.join(pivot_dir, 'pivot.npy')):
                    continue
                img_dir1 = os.path.join(axis_dir, 'imgs')
                img_dir2 = os.path.join(pivot_dir, 'imgs')
                if os.path.exists(img_dir1):
                    shutil.rmtree(img_dir1)
                if os.path.exists(img_dir2):
                    shutil.rmtree(img_dir2)
                os.makedirs(axis_dir, exist_ok=True)
                os.makedirs(pivot_dir, exist_ok=True)

                child_mesh, other_mesh, judge_flag = judge_child_parent_contact(part_seg_dir, entity,
                                                                                child_label,
                                                                                parent_label)
                # if child_mesh is None:
                #     continue
                print(judge_flag)
                if not judge_flag:
                    child_mesh, other_mesh = get_child_parent_mesh(part_seg_dir, entity, child_label)
                newplane_fine, centers, plane_point = get_candidate_axes_for_revolute(child_mesh,
                                                                                      other_mesh)
                # if newplane_fine is None and plane_point is None:
                #     continue
                target_bbox = child_mesh.bounds

                # 保存候选轴
                for i in range(len(newplane_fine)):
                    arrow_mesh, rotation = create_3d_arrow_mesh(
                        position=plane_point + 0.04 * newplane_fine[0][:3] / np.linalg.norm(
                            newplane_fine[0][:3]), direction=newplane_fine[i][:3], color=color_list[i],
                        target_bbox=child_mesh.bounds)

                    arrow_mesh.export(os.path.join(axis_dir, f'axis_{i + 1}.obj'))

                # 保存候选位置点 pivot point
                pivot_points = np.concatenate([plane_point[None], centers], axis=0)
                diag_len = np.linalg.norm(target_bbox[1] - target_bbox[0])
                for c_id in range(len(pivot_points)):
                    # print(diag_len)
                    point_mesh = trimesh.creation.icosphere(radius=diag_len * 0.06, subdivisions=3)
                    point_mesh.apply_translation(pivot_points[c_id])
                    point_mesh.visual.face_colors = color_list[c_id]
                    point_mesh.export(os.path.join(pivot_dir, f'point_{c_id + 1}.obj'))

                axis_part_save_pth = os.path.join(axis_dir, 'part.obj')
                pivot_part_save_pth = os.path.join(pivot_dir, 'part.obj')

                child_mesh.export(axis_part_save_pth)
                clean_obj_id(axis_part_save_pth)

                child_mesh.export(pivot_part_save_pth)
                clean_obj_id(pivot_part_save_pth)

                np.save(os.path.join(axis_dir, 'axis.npy'), np.stack(newplane_fine)[:, :3])
                np.save(os.path.join(pivot_dir, 'pivot.npy'), pivot_points)
                # np.save(os.path.join(rotate_dir, entity, str(child_label), 'axis.npy'),
                #         np.concatenate([plane_point[None], np.stack(newplane_fine)[:, :3], centers], 0))

