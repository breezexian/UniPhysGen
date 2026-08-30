import trimesh
import numpy as np
import random


def read_mesh(pth):
    mesh_or_scene = trimesh.load(pth, process=False)
    # handle Scene or Mesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        #mesh = mesh_or_scene.dump(concatenate=True)  # 合并所有 mesh
        mesh = mesh_or_scene.to_geometry()
    else:
        mesh = mesh_or_scene
    return mesh


def face_areas(vertices, faces):
    """
    计算 mesh 每个三角面的面积
    vertices: (N, 3) 顶点坐标
    faces: (M, 3) 三角面顶点索引
    return: (M,) 面积数组
    """
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


def get_face_areas_list(pth):
    mesh = read_mesh(pth)
    vertices = mesh.vertices
    faces = mesh.faces
    areas = face_areas(vertices, faces)
    return areas


def build_mesh_from_part(mesh, t_faces, sam_part=False):
    mesh = mesh.copy()
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()
    F = len(mesh.faces)
    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)  # (3F,3)
    new_faces = np.arange(len(new_vertices), dtype=np.int64).reshape(-1, 3)
    face_rgb = []
    for ind, face in enumerate(mesh.faces):
        face_id = ind
        if sam_part:
            face_id = ind + 1
        if face_id not in t_faces:
            face_rgb.append([128, 128, 128])
            continue
        face_rgb.append([255, 0, 0])

    vertex_colors = np.repeat(face_rgb, 3, axis=0)  # (3F,3)
    alpha = np.ones((vertex_colors.shape[0], 1), dtype=np.uint8) * 255
    vertex_colors_rgba = np.concatenate([vertex_colors, alpha], axis=1)

    new_mesh = trimesh.Trimesh(vertices=new_vertices,
                               faces=new_faces,
                               vertex_colors=vertex_colors_rgba,
                               process=False)
    return new_mesh


def build_mesh_from_face2cls(mesh, face2cls):
    mesh = mesh.copy()
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()
    F = len(mesh.faces)
    new_vertices = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)  # (3F,3)
    new_faces = np.arange(len(new_vertices), dtype=np.int64).reshape(-1, 3)
    face_rgb = []
    cls_set = set(face2cls.values())
    cls2col = {}
    for gid in cls_set:
        if gid not in cls2col:
            cls2col[gid] = [random.randint(0, 255) for i in range(3)]
    for ind, face in enumerate(mesh.faces):
        face_id = str(ind)
        if face_id not in face2cls:
            face_rgb.append([255, 0, 0])
            continue
        face_rgb.append(cls2col[face2cls[face_id]])

    vertex_colors = np.repeat(face_rgb, 3, axis=0)  # (3F,3)
    alpha = np.ones((vertex_colors.shape[0], 1), dtype=np.uint8) * 255
    vertex_colors_rgba = np.concatenate([vertex_colors, alpha], axis=1)

    new_mesh = trimesh.Trimesh(vertices=new_vertices,
                               faces=new_faces,
                               vertex_colors=vertex_colors_rgba,
                               process=False)
    return new_mesh
