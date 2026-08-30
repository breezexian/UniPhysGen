import trimesh
import json
import os
import numpy as np
from trimesh.visual.texture import TextureVisuals
from PIL import Image


def glb_to_part_main(glb_path, face2mlt_json, output_dir):
    """
    将一个带纹理的 GLB 模型按 face 部件划分拆成多个 obj（保持原始纹理）
    :param glb_path: 输入的 glb 文件路径
    :param face2mlt_json: face->part 对应关系的 json 文件路径
    :param output_dir: 输出文件夹
    """
    os.makedirs(os.path.join(output_dir, "glbs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "objs"), exist_ok=True)

    # 1. 读取 mesh
    mesh = trimesh.load(glb_path, force='mesh')

    # 2. 读取 face2mlt
    with open(face2mlt_json, 'r') as f:
        face2mlt = json.load(f)

    # 3. 按 part_id 分组 faces
    part_to_faces = {}
    for face_id_str, part_id in face2mlt.items():
        face_id = int(face_id_str)
        part_to_faces.setdefault(part_id, []).append(face_id)

    # 4. 遍历每个部件，提取子 mesh
    idx_name = 1
    for part_id, face_inds in part_to_faces.items():
        sub_faces = mesh.faces[face_inds]
        # 找到子 mesh 涉及的顶点
        unique_vertices, inverse_indices = np.unique(sub_faces, return_inverse=True)
        sub_vertices = mesh.vertices[unique_vertices]
        sub_faces = inverse_indices.reshape(-1, 3)

        # 纹理信息

        sub_uvs = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            if mesh.visual.uv is not None:
                sub_uvs = mesh.visual.uv[unique_vertices]

        sub_mesh = trimesh.Trimesh(
            vertices=sub_vertices,
            faces=sub_faces,
            process=False
        )

        # 保留材质和纹理
        if hasattr(mesh.visual, 'material'):
            sub_mesh.visual.material = mesh.visual.material

        # 保留 UV（纹理坐标）
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            sub_mesh.visual.uv = mesh.visual.uv[unique_vertices]

        # 5. 导出 OBJ
        out_path = os.path.join(output_dir, "objs", f"{idx_name}.obj")
        sub_mesh.export(out_path)
        print(f"✅ Saved part {part_id} -> {out_path}")

        # 导出 GLB
        out_path = os.path.join(output_dir, "glbs", f"{idx_name}.glb")
        sub_mesh.export(out_path)
        print(f"✅ Saved part {part_id} -> {out_path}")

        idx_name += 1

    print("全部部件已成功导出。")
