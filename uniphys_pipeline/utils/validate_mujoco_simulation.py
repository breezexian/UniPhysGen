import mujoco
import numpy as np
import trimesh
import torch
import open3d as o3d
import mujoco.viewer
import time
import copy
import os
import cv2
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree
import argparse
from subprocess import call, check_call


# ==========================
# 函数定义
# ==========================

def geom_to_trimesh(model, data, geom_id):
    """从 Mujoco geom 构建 trimesh.Mesh（世界坐标下）"""
    mesh_id = model.geom_dataid[geom_id]
    verts = model.mesh_vert[model.mesh_vertadr[mesh_id]:
                            model.mesh_vertadr[mesh_id] + 3 * model.mesh_vertnum[mesh_id]]
    verts = verts.reshape(-1, 3).copy()
    faces = model.mesh_face[model.mesh_faceadr[mesh_id]:
                            model.mesh_faceadr[mesh_id] + 3 * model.mesh_facenum[mesh_id]]
    faces = faces.reshape(-1, 3).copy()
    pos = data.geom_xpos[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    verts = verts @ mat.T + pos
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def geom_to_trimesh_safe(model, data, geom_id):
    """安全地将任意 geom 转为 trimesh.Mesh，世界坐标下"""
    gtype = model.geom_type[geom_id]
    gsize = model.geom_size[geom_id]
    gpos = data.geom_xpos[geom_id]
    grot = data.geom_xmat[geom_id].reshape(3, 3)

    mesh = None

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            return None
        verts = model.mesh_vert[
                model.mesh_vertadr[mesh_id]:
                model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
                ]
        faces = model.mesh_face[
                model.mesh_faceadr[mesh_id]:
                model.mesh_faceadr[mesh_id] + model.mesh_facenum[mesh_id]
                ]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * gsize)

    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=gsize[0])

    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=gsize[0], height=2 * gsize[1])

    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=gsize[0], height=2 * gsize[1])

    else:
        print(f"⚠️ geom {geom_id} 类型 {gtype} 暂不支持")
        return None

    # 转为世界坐标
    mesh.apply_transform(np.vstack([
        np.hstack([grot, gpos.reshape(3, 1)]),
        [0, 0, 0, 1]
    ]))
    return mesh


def get_body_geom_ids(model, body_name):
    """获取某个 body 下所有 geom id"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    start = model.body_geomadr[body_id]
    count = model.body_geomnum[body_id]
    return list(range(start, start + count))


def get_body_and_children_geom_ids(model, body_name):
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    assert body_id >= 0, f"Body {body_name} not found"

    geom_ids = []

    def dfs(bid):
        # 当前 body 的 geom
        start = model.body_geomadr[bid]
        count = model.body_geomnum[bid]
        geom_ids.extend(range(start, start + count))

        # 遍历所有 body，找子 body
        for child_id in range(model.nbody):
            if model.body_parentid[child_id] == bid:
                dfs(child_id)

    dfs(body_id)
    return geom_ids


def get_all_child_bodies(model, parent_body_id):
    """递归获取 parent_body_id 下所有子 body 的 id"""
    child_ids = []
    for i in range(model.nbody):
        if model.body_parentid[i] == parent_body_id:
            child_ids.append(i)
            child_ids.extend(get_all_child_bodies(model, i))
    return child_ids


def make_combined_mesh(model, data, geom_ids=[]):
    """合并多个静态 geom 成一个 trimesh"""
    meshes = []
    for gid in geom_ids:
        mesh = geom_to_trimesh_safe(model, data, gid)
        if mesh is None:
            return None
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


# def get_contact_region(child_mesh, other_mesh, threshold=1e-2,
#                        child_sample=1000, other_sample=1000,
#                        max_count=1000, device_id=0):
#     """基于 mesh 采样点求接触区域"""
#     device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
#
#     child_pts = np.array(child_mesh.vertices)
#     if child_sample > 0:
#         extra = np.array(child_mesh.sample(child_sample))
#         child_pts = np.vstack([child_pts, extra])
#     child_pts = torch.tensor(child_pts, dtype=torch.float32, device=device)
#
#     other_pts = np.array(other_mesh.vertices)
#     if other_sample > 0:
#         extra = np.array(other_mesh.sample(other_sample))
#         other_pts = np.vstack([other_pts, extra])
#     other_pts = torch.tensor(other_pts, dtype=torch.float32, device=device)
#
#     diff = child_pts[:, None, :] - other_pts[None, :, :]
#     dists = torch.norm(diff, dim=-1)
#
#     count = 0
#     jointregion = torch.zeros((0, 3), device=device)
#
#     while count < max_count:
#         mask = dists.min(dim=0).values < threshold
#         jointregion = other_pts[mask]
#         if len(jointregion) > 6:
#             break
#         threshold += 1e-2
#         count += 1
#
#     return jointregion.cpu().numpy(), threshold

@torch.no_grad()
def get_contact_region(child_mesh, other_mesh, threshold=1e-2, oversize=False, use_gpu=True, scale=2,
                       child_sample=1000, other_sample=1000,
                       max_count=1000, device_id=0):
    """
    基于 mesh surface 采样，GPU 并行计算接触区域
    保留原顶点 + 采样额外点
    当接触点不足时，逐步增加阈值
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if not use_gpu:
        device = "cpu"
    # device = "cpu"
    # child mesh 顶点 + 采样
    child_pts = np.array(child_mesh.vertices)
    if oversize:
        random_indices = np.random.choice(len(child_pts), size=len(child_pts) // scale, replace=False)
        child_pts = child_pts[random_indices]
    if child_sample > 0:
        extra_child_pts = np.array(child_mesh.sample(child_sample))
        child_pts = np.vstack([child_pts, extra_child_pts])
    child_pts = torch.tensor(child_pts, dtype=torch.float32, device=device)
    # other mesh 顶点 + 采样
    other_pts = np.array(other_mesh.vertices)
    if oversize:
        random_indices = np.random.choice(len(other_pts), size=len(other_pts) // scale, replace=False)
        other_pts = other_pts[random_indices]
    if other_sample > 0:
        extra_other_pts = np.array(other_mesh.sample(other_sample))
        other_pts = np.vstack([other_pts, extra_other_pts])

    other_pts = torch.tensor(other_pts, dtype=torch.float32, device=device)

    # 计算 pairwise 距离
    diff = child_pts[:, None, :] - other_pts[None, :, :]
    dists = torch.norm(diff, dim=-1)  # shape: [N_child, N_other]

    count = 0
    jointregion = torch.zeros((0, 3), device=device)

    while count < max_count:
        mask = dists.min(dim=0).values < threshold
        jointregion = other_pts[mask]
        if len(jointregion) > 6:  # 足够多的点
            break
        threshold += 1e-2
        count += 1
    del diff
    del dists
    del child_pts
    del other_pts
    if not use_gpu:
        return jointregion.numpy(), threshold
    return jointregion.cpu().numpy(), threshold


# points_world: Nx3，points所属geom的geom_id
def update_points_by_geom(points_world, geom_ids, model, data):
    updated_points = []
    for p, gid in zip(points_world, geom_ids):
        # geom世界变换
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        # 将初始点从世界坐标映射到 geom 局部，再映射到当前帧世界坐标
        p_local = np.linalg.solve(mat, (p - pos))  # x_local = mat^-1 @ (x - pos)
        p_world_new = p_local @ mat.T + pos  # 当前帧世界坐标
        updated_points.append(p_world_new)
    return np.array(updated_points)


def fit_plane(points):
    """最小二乘拟合平面 Ax+By+Cz+D=0，返回单位法向量n及D"""
    centroid = np.mean(points, axis=0)
    uu, dd, vv = np.linalg.svd(points - centroid)
    normal = vv[2, :]
    normal = normal / np.linalg.norm(normal)
    D = -np.dot(normal, centroid)
    return normal, D


def fit_plane_ransac(pcd, distance_threshold=0.01, max_planes=5, min_inliers=100):
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


def get_plane_info(contact_pts):
    pcd = o3d.geometry.PointCloud()
    nn_points = copy.deepcopy(contact_pts)
    nn_points = nn_points.astype(np.float64)  # 强制转 float64 才可以进行o3d的操作
    pcd.points = o3d.utility.Vector3dVector(nn_points)
    planes, innerpoint = fit_plane_ransac(pcd, distance_threshold=0.05, max_planes=1,
                                          min_inliers=int(len(contact_pts) * 0.1))
    para1, para2, para3, para4 = planes[0]
    if len(contact_pts[innerpoint[0]]) > 0:
        plane_point = contact_pts[innerpoint[0]].mean(0)
    else:
        plane_point = contact_pts.mean(0)
    normal = np.array([para1, para2, para3])
    scale = np.linalg.norm(normal)
    return normal / scale, para4 / scale, plane_point


def plane_angle_and_distance(n1, D1, n2, D2):
    """计算两个平面夹角(°)与距离差"""
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    dist_diff = abs(np.dot(-n1 * D1, n2) + D2)
    return angle, dist_diff


def compute_thickness(points, n, D):
    proj = points @ n + D  # np.dot(contact_pts, n0) + D0
    return proj.max() - proj.min()


def contact_overlap_ratio(P0, P1, eps=0.01):
    if len(P0) == 0 or len(P1) == 0:
        return 0.0

    tree0 = cKDTree(P0)
    tree1 = cKDTree(P1)

    r1 = np.mean(tree0.query(P1)[0] < eps)
    r0 = np.mean(tree1.query(P0)[0] < eps)

    return r1  # min(r0, r1)


def compute_mean_nn_distance(points: np.ndarray) -> float:
    """
    points: (N, 3) numpy array
    return: mean nearest neighbor distance (excluding self)
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) >= 2, "Need at least 2 points"

    tree = cKDTree(points)
    # k=2: 第一个是自己 (dist=0)，第二个是真正最近邻
    dists, _ = tree.query(points, k=2, workers=-1)

    nn_dists = dists[:, 1]
    return float(np.mean(nn_dists))


def validate(pth, save_dir, enable_render, tot_steps=200, reverse=False):
    #try:
    video_writer = None
    renderer = None
    cam_scale = 2.2
    if "partnet" in save_dir:
        cam_scale = 2.5
    print(pth, save_dir, "cam_scale:", cam_scale)
    # ==========================
    # 初始化模型
    # ==========================
    model = mujoco.MjModel.from_xml_path(pth)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 运动部件 body 名称
    motion_body_name = "motion_part"
    base_body_name = "part_base"
    motion_geom_ids = get_body_geom_ids(model, motion_body_name)
    # motion_geom_ids = get_body_and_children_geom_ids(model, motion_body_name)
    base_geom_ids = get_body_geom_ids(model, base_body_name)
    static_mesh = make_combined_mesh(model, data, base_geom_ids)
    # static_mesh.show()
    # ==========================
    # 初始状态计算初始平面
    # ==========================
    motion_mesh = make_combined_mesh(model, data, motion_geom_ids)
    if motion_mesh is None:
        return -1, [], {}
    # motion_mesh.show()
    try:
        ori_contact_pts, ori_th = get_contact_region(motion_mesh, static_mesh)
    except Exception as e:
        try:
            print("Contact region compute failed!", e)
            ori_contact_pts, ori_th = get_contact_region(motion_mesh, static_mesh, oversize=True)
        except Exception as e:
            print("Contact region compute failed again! try the last time in cpu!", e)
            ori_contact_pts, ori_th = get_contact_region(motion_mesh, static_mesh, oversize=True, use_gpu=False, scale=4)
    print(ori_contact_pts.shape)
    # ori_center = np.mean(ori_contact_pts, axis=0)
    if len(ori_contact_pts) < 6:
        print("初始状态无明显接触点")
        return -1, [], {}
    else:
        n0, D0, ori_center = get_plane_info(ori_contact_pts)
        thickness_ori = compute_thickness(ori_contact_pts, n0, D0)
        thickness_ori = max(thickness_ori, 1e-4)
        ori_pts_dist_mean = compute_mean_nn_distance(ori_contact_pts)
        # n0, D0 = fit_plane(ori_contact_pts)
        print("初始平面法线:", n0, "平面中心:", ori_center)

    # ==========================
    # 仿真过程检测每帧接触平面变化
    # ==========================
    jnt_name = "joint_1"
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    jnt_type = model.jnt_type[jnt_id]
    jnt_pos = model.jnt_pos[jnt_id]  # shape (3,)
    jnt_axis = model.jnt_axis[jnt_id]  # shape (3,)
    print(jnt_type, jnt_axis, jnt_pos)
    qpos_min, qpos_max = model.jnt_range[jnt_id]
    ori_qpos_min = qpos_min
    ori_qpos_max = qpos_max
    if reverse:
        qpos_min = -1 * qpos_min
        qpos_max = -1 * qpos_max
    print(f"Joint range: {qpos_min:.2f} ~ {qpos_max:.2f} rad")
    # 初始化关节位置
    data.qpos[jnt_id] = qpos_min

    mujoco.mj_forward(model, data)

    # -------------------------
    # 轨迹
    traj = np.linspace(qpos_min, qpos_max, tot_steps)
    if enable_render:
        # -------------------------
        # 启动交互窗口
        # viewer = mujoco.viewer.launch_passive(model, data)

        # 使用 framergb 设置背景颜色 (RGB格式，没有alpha通道)
        # 初始化 Renderer
        render_width, render_height = 1000, 1000
        renderer = mujoco.Renderer(model, render_width, render_height)

        # 渲染第一帧获取实际尺寸
        rgb = renderer.render()
        render_height, render_width, _ = rgb.shape

        # 初始化视频写入器
        fps = 60
        sname = os.path.basename(pth).split(".")[0]
        video_writer = cv2.VideoWriter(os.path.join(save_dir, f"{sname}_tmp.mp4"),
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (render_width, render_height))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        geom_pos = np.array([model.geom_pos[i] for i in range(model.ngeom) if i != floor_id])
        geom_size = np.array([model.geom_size[i] for i in range(model.ngeom) if i != floor_id])

        min_corner = (geom_pos - geom_size).min(axis=0)
        max_corner = (geom_pos + geom_size).max(axis=0)

        lookat = (min_corner + max_corner) / 2
        # 最大边长（用于计算相机距离）
        bbox_size = (max_corner - min_corner).max()

        print(lookat)

    print("initial_contact_points：", len(ori_contact_pts), "initial_threshold：", ori_th)
    out_range_degrees = []
    out_range_dists = []
    out_center_dists = []

    all_angles = []
    all_dists = []
    all_center_dists = []
    all_thickness_ratios = []
    all_overlap_ratios = []
    no_contact = False
    for step, q in enumerate(traj):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        if enable_render:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = lookat
            cam.lookat[:] = lookat  # 看向物体中心
            cam.distance = bbox_size / (2 * np.tan(np.deg2rad(45 / 2))) * cam_scale
            try:
                renderer.update_scene(data, camera=cam)  # free 相机防止不存在报错
                rgb = renderer.render()
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            except Exception as e:
                print("渲染失败:", e)
        motion_mesh = make_combined_mesh(model, data, motion_geom_ids)
        if motion_mesh is None:
            return -1, [], {}
        try:
            contact_pts, th = get_contact_region(motion_mesh, static_mesh, threshold=ori_th)
        except Exception as e:
            #print("Steps: contact region compute failed!")
            try:
                contact_pts, th = get_contact_region(motion_mesh, static_mesh, threshold=ori_th, oversize=True)
            except Exception as e:
                #print("Steps: Contact region compute failed again! try the last time in cpu!", e)
                contact_pts, th = get_contact_region(motion_mesh, static_mesh, threshold=ori_th, oversize=True, use_gpu=False, scale=4)
        if th > ori_th + 0.05 or len(contact_pts) < 6:
            # or (jnt_type == mujoco.mjtJoint.mjJNT_HINGE and th > 2 * ori_th):
            # 距离太大或者需要超过原来的阈值才能有满足条件的接触点时认为无接触点
            no_contact = True
            # print("无接触", th, ori_th)
            # print(f"q={q:.3f}: 无接触点")
            continue
        # n1, D1 = fit_plane(contact_pts)
        # cur_center = np.mean(contact_pts, axis=0)
        n1, D1, cur_center = get_plane_info(contact_pts)
        cur_center_dist = np.linalg.norm(cur_center - ori_center)

        # align the direction
        if np.dot(n0, n1) < 0:
            n1 = -n1
            D1 = -D1
        angle, dist = plane_angle_and_distance(n0, D0, n1, D1)
        # angle2, dist2 = plane_angle_and_distance(n0, D0, -n1/np.linalg.norm(-n1), -D1/np.linalg.norm(-n1))
        all_angles.append(angle)
        all_dists.append(dist)
        all_center_dists.append(cur_center_dist)

        if angle > 15:
            out_range_degrees.append(angle)
            # print(f"异常运动：平面变化过大 ({angle:.2f}°), 距离差=（{dist:.4f}）")
        if dist > 0.05:
            out_range_dists.append(dist)

        if cur_center_dist > 0.05:
            out_center_dists.append(cur_center_dist)

        overlap_ratio = contact_overlap_ratio(ori_contact_pts, contact_pts, eps=ori_pts_dist_mean * 2)
        all_overlap_ratios.append(overlap_ratio)
        thickness_cur = compute_thickness(contact_pts, n0, D0)
        thickness_ratio = thickness_cur / thickness_ori
        all_thickness_ratios.append(thickness_ratio)

        if step % 20 == 0:  # or step > 180:
            print(
                f"step={step}: 平面夹角={angle:.2f}°, 距离差={cur_center_dist:.4f}, thickness={max(all_thickness_ratios)}, {overlap_ratio}, {ori_pts_dist_mean}")
        # print(len(contact_pts), "th：", th)
        # 更新窗口
        # viewer.sync()
        # time.sleep(0.01)

    print("检测完毕。")
    tot_angle_mean = (np.mean(all_angles) if len(all_angles) else 0)  # 全局扰动强度
    out_angle_mean = (np.mean(out_range_degrees) if len(out_range_degrees) else 0)  # 异常帧严重程度

    tot_dist_mean = (np.mean(all_dists) if len(all_dists) else 0)  # 全局扰动强度
    out_dist_mean = (np.mean(out_range_dists) if len(out_range_dists) else 0)  # 异常帧严重程度

    tot_center_mean = (np.mean(all_center_dists) if len(all_center_dists) else 0)  # 全局扰动强度
    out_center_mean = (np.mean(out_center_dists) if len(out_center_dists) else 0)  # 异常帧严重程度

    out_angle_ratio = len(out_range_degrees) / tot_steps
    out_center_ratio = len(out_center_dists) / tot_steps
    tot_ratio = len(all_angles) / tot_steps

    thickness_ratio_percentile = (np.percentile(all_thickness_ratios, 95) if len(all_thickness_ratios) > 0 else 0)
    overlap_ratio_percentile = (np.percentile(all_overlap_ratios, 10) if len(all_overlap_ratios) > 0 else 1)
    bad_overlap_ratio = (np.mean(np.array(all_overlap_ratios) < 0.4) if len(all_overlap_ratios) > 0 else 1)
    bad_thickness_num = (np.sum(np.array(all_thickness_ratios) > 3) if len(all_thickness_ratios) > 0 else 0)
    bad_thickness_ratio = (np.mean(np.array(all_thickness_ratios) > 3) if len(all_thickness_ratios) > 0 else 1)
    print("thickness_ratio_percentile", thickness_ratio_percentile)
    print("overlap_ratio_percentile", overlap_ratio_percentile)
    print("bad_overlap_ratio", bad_overlap_ratio)
    metrics_dict = {
        'tot_angle_mean': float(tot_angle_mean),
        'out_angle_mean': float(out_angle_mean),
        'tot_dist_mean': float(tot_dist_mean),
        'out_dist_mean': float(out_dist_mean),
        'tot_center_mean': float(tot_center_mean),
        'out_center_mean': float(out_center_mean),
        'out_angle_ratio': float(out_angle_ratio),
        'out_center_ratio': float(out_center_ratio),
        'tot_ratio': float(tot_ratio),
        'bad_overlap_ratio': float(bad_overlap_ratio),
        'thickness_ratio_percentile': float(thickness_ratio_percentile),
        'overlap_ratio_percentile': float(overlap_ratio_percentile),
        'bad_thickness_num': float(bad_thickness_num),
        'bad_thickness_ratio': float(bad_thickness_ratio)
    }

    metrics_dict.update({
        'total_steps': int(tot_steps),  # 步数应该是整数
        'num_all_angles': int(len(all_angles)),
        'num_out_range_degrees': int(len(out_range_degrees)),
        'num_all_dists': int(len(all_dists)),
        'num_out_range_dists': int(len(out_range_dists)),
        'num_all_center_dists': int(len(all_center_dists)),
        'num_out_center_dists': int(len(out_center_dists)),
        'num_thickness_ratios': int(len(all_thickness_ratios)),
        'num_overlap_ratios': int(len(all_overlap_ratios)),
    })

    motion_info = {
        "jnt_type": int(jnt_type),
        "jnt_axis": list(np.array(jnt_axis).astype(float)),
        "jnt_pos": list(np.array(jnt_pos).astype(float)),
        "jnt_range": list(np.array([ori_qpos_min, ori_qpos_max]).astype(float)),
    }

    validate_info = {
        "metrics": metrics_dict,
        "motion_info": motion_info
    }
    level = None
    reasons = []
    # ---------- Level-0: Hard Invalid  Contact Identity & Feasibility----------
    # Contact Validity (Hard Constraint)
    # 1. 根本没接触了 2. 明显穿模 / 互相穿透 3. 接触面被整体换掉
    if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
        if no_contact:
            reason = "HINGE joint lost contact"
            reasons.append(reason)
        if bad_overlap_ratio > 0.1 and overlap_ratio_percentile < 0.4:
            reason = f"Severe overlap_ratio change: p10={overlap_ratio_percentile:.2f}, ratio={bad_overlap_ratio:.2f}"
            reasons.append(reason)
    # if jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
    #     if tot_ratio > 0.1 and tot_angle_mean > 20:
    #         reason = f"SLIDE joint lost plane. angle = {tot_angle_mean}"
    #         reasons.append(reason)
    if thickness_ratio_percentile > 3:
        reason = f"Severe thickness change (p95={thickness_ratio_percentile:.2f})"
        reasons.append(reason)

    if len(reasons) > 0:
        level = 0
    else:
        level = 1
        reasons.append("success")
    if len(all_angles) == 0:
        level = -2
        reasons = ["no info!"]
    return level, reasons, validate_info
    # except Exception as e:
    #     print(e)
    #     level = -3
    #     reasons = [str(e)]
    #     return level, reasons, {}
    # finally:
    #     torch.cuda.empty_cache()
    #
    #     if video_writer is not None:
    #         video_writer.release()
    #     if renderer is not None:
    #         del renderer


def reverse_range(reasons, bad_thickness_ratio):
    for reason in reasons:
        if "thickness" in reason and bad_thickness_ratio > 0.5:
            return True
    return False


def process_validation(pth, save_dir, enable_render=False):
    sname = os.path.basename(pth).split(".")[0]
    res_file = os.path.join(save_dir, f"{sname}.json")

    if os.path.exists(res_file):
        return "completed, skip!"
    level, reasons, validate_info = validate(pth, save_dir, enable_render)
    flag_reverse = False
    if "motion_info" in validate_info:
        jnt_type = validate_info["motion_info"]["jnt_type"]
        jnt_range = validate_info["motion_info"]["jnt_range"]
        bad_thickness_ratio = validate_info["metrics"]["bad_thickness_ratio"]
        print(jnt_type, jnt_range, bad_thickness_ratio)
        if (jnt_type == 2 or (jnt_type == 3 and jnt_range[-1] < 6.28)) and level == 0 \
                and reverse_range(reasons, bad_thickness_ratio):
            print("===reverse===")
            flag_reverse = True
            level, reasons, validate_info = validate(pth, save_dir, enable_render, reverse=True)
    print("level:", level, "reasons:", reasons, "reverse", flag_reverse)
    res = {"level": level, "reasons": reasons, "reverse": flag_reverse, "validate_info": validate_info}
    json.dump(res, open(res_file, "w"), indent=4)
    # tmp_video_file = os.path.join(save_dir, f"{sname}_tmp.mp4")
    # new_video_file = os.path.join(save_dir, f"{sname}.mp4")
    # if os.path.exists(tmp_video_file):
    #     cmd = f"ffmpeg -y -i {tmp_video_file} -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k {new_video_file}"
    #     check_call(cmd, shell=True)
    #     cmd = f"rm -f {tmp_video_file}"
    #     check_call(cmd, shell=True)
    return


def mujoco_sim_validation_main(gpt_output, entity, enable_render=True):
    entity_dir = os.path.join(gpt_output, "kg_xml", entity)
    # print(entity_dir)
    if not os.path.exists(entity_dir):
        return
    for xml_file in os.listdir(entity_dir):
        if not xml_file.endswith(".xml"):
            continue
        xml_pth = os.path.join(entity_dir, xml_file)
        process_validation(xml_pth, entity_dir, enable_render)
