import bpy
import os
import mathutils
import json
import math
import numpy as np
import sys


def rotate_all_mesh_geometry_x_minus_90():
    R = mathutils.Matrix.Rotation(-math.pi / 2, 4, 'X')

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.transform(R)
            obj.data.update()


def render_part_ori(obj_paths, output_dir):
    """
    渲染多个部件OBJ文件，输出每个像素对应的部件ID
    
    Args:
        obj_paths: 部件OBJ文件路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    render_width = 800
    render_height = 800

    pitch_z = 0.7
    cam_positions = {
        # 'view_0': [1, 1, pitch_z],
        # 'view_1': [-1, 1, pitch_z],
        # 'view_2': [1, 1, -pitch_z],
        # 'view_3': [-1, 1, -pitch_z],
        'view_4': [1, -1, pitch_z],
        # 'view_5': [-1, -1, pitch_z],
        # 'view_6': [1, -1, -pitch_z],
        # 'view_7': [-1, -1, -pitch_z],
    }

    # -------------------------
    # 清空场景
    # -------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # -------------------------
    # 导入所有部件OBJ并分配材质ID
    # -------------------------
    part_id_dict = {}  # 部件名 -> 材质ID
    all_mesh_objs = []

    for idx, obj_path in enumerate(obj_paths):
        if not os.path.exists(obj_path):
            print(f"警告: 文件不存在 {obj_path}")
            continue

        # 获取部件名（文件名去掉扩展名）
        part_name = os.path.basename(obj_path).split(".")[0]
        part_id = int(part_name)  # 从1开始，0通常表示背景

        # 导入OBJ
        bpy.ops.wm.obj_import(filepath=obj_path)

        # 获取刚导入的对象
        imported_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']

        for obj in imported_objs:
            all_mesh_objs.append(obj)

    if len(all_mesh_objs) == 0:
        raise ValueError("没有成功导入任何Mesh对象！")

    # -------------------------
    # 渲染设置
    # -------------------------
    scene = bpy.context.scene
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.engine = 'CYCLES'  # 或 'BLENDER_EEVEE'
    scene.cycles.device = 'GPU'  # 启用 GPU 渲染
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()
    prefs.compute_device_type = "CUDA"  # prefs.compute_device_type = 'OPTIX'
    # 启用所有可用的 GPU
    # prefs.get_devices()
    # for d in prefs.devices:
    #     if 'NVIDIA' in d.name:
    #         print(d.name)
    #         d.use = True
    #     else:
    #         d.use = False
    scene.cycles.samples = 128
    scene.render.film_transparent = True  # 背景透明，可改成 False

    # 创建或获取 World
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)  # 白色
    bg.inputs[1].default_value = 1.0  # 强度

    # -------------------------
    # 创建相机
    # -------------------------
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    def look_at(cam, target):
        direction = target - cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

    # -------------------------
    # 计算所有部件的整体包围盒中心
    # -------------------------
    all_bbox_corners = []
    for obj in all_mesh_objs:
        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        all_bbox_corners.extend(bbox)

    # 计算整体中心和半径
    min_corner = mathutils.Vector((
        min(v.x for v in all_bbox_corners),
        min(v.y for v in all_bbox_corners),
        min(v.z for v in all_bbox_corners)
    ))
    max_corner = mathutils.Vector((
        max(v.x for v in all_bbox_corners),
        max(v.y for v in all_bbox_corners),
        max(v.z for v in all_bbox_corners)
    ))
    center = (min_corner + max_corner) / 2.0
    radius = (max_corner - min_corner).length / 2.0
    print("物体整体中心:", center, "半径:", radius)

    # 相机到物体中心的合理距离（保证物体完全进入视野）
    fov = cam_data.angle
    dist = radius / math.tan(fov / 2) * 1.5  # 乘个安全系数

    # -------------------------
    # 渲染循环
    # -------------------------
    for name, pos in cam_positions.items():
        direction = mathutils.Vector(pos).normalized()
        cam_obj.location = center + direction * dist
        look_at(cam_obj, center)
        # 输出路径
        scene.render.filepath = os.path.join(output_dir, f"{name}.png")
        # 渲染
        bpy.ops.render.render(write_still=True)

    print("Part ID Pass (EXR) 输出完成:", output_dir)


def read_part_id_exr(exr_path):
    """
    读取EXR文件中的部件ID
    
    Args:
        exr_path: EXR文件路径
    
    Returns:
        numpy数组，每个像素的部件ID（整数）
    """
    import OpenEXR
    import Imath

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取R通道（Material Index存储在R通道）
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel('R', pt)
    r_data = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)

    # 转换为整数ID
    part_ids = r_data.astype(np.int32)

    return part_ids


if __name__ == '__main__':

    try:
        print(sys.argv)
        obj_dir = sys.argv[5]
        output_dir = sys.argv[6]
        obj_paths = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith(".obj")]
        render_part_ori(obj_paths, output_dir)
    except Exception as e:
        print(f"Blender 内部错误: {e}")
        sys.exit(1)
