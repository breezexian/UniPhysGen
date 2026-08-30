import bpy
import os
import mathutils
import json
import math
import numpy as np
import sys


def render_face(glb_path, output_dir):
    # -------------------------
    # 参数设置
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)

    render_width = 800
    render_height = 800

    pitch_z = 0.7
    cam_positions = {
        'view_0': [1, 1, pitch_z],
        'view_1': [-1, 1, pitch_z],
        'view_2': [1, 1, -pitch_z],
        'view_3': [-1, 1, -pitch_z],
        'view_4': [1, -1, pitch_z],
        'view_5': [-1, -1, pitch_z],
        'view_6': [1, -1, -pitch_z],
        'view_7': [-1, -1, -pitch_z],
    }
    # -------------------------
    # 清空场景
    # -------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # -------------------------
    # 导入 glb
    # -------------------------
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # 找到第一个 Mesh 对象
    mesh_objs = [o for o in bpy.context.scene.collection.objects if o.type == 'MESH']
    if len(mesh_objs) == 0:
        raise ValueError("没有找到 Mesh 对象，请检查 glb 文件！")
    obj = mesh_objs[0]
    print("找到 Mesh 对象:", obj.name)

    # -------------------------
    # 给每个面分配唯一材质（方便 Face Pass）
    # -------------------------
    face_id_dict = {}
    mesh = obj.data
    mesh.materials.clear()

    print("面片数", len(mesh.polygons))
    print("顶点数 (vertices):", len(mesh.vertices))
    # if len(mesh.polygons) > 50000:
    #     print("error", glb_path)
    #     return

    for i, poly in enumerate(mesh.polygons):
        # 创建一个新材质
        mat = bpy.data.materials.new(name=f"Mat_{i}")
        mat.pass_index = i + 1  # 每个面唯一 ID
        mesh.materials.append(mat)
        poly.material_index = i  # 绑定到面

        vert_key = list(map(str, list(poly.vertices)))  # 顶点索引list
        # mat_id = mesh.materials[f"Mat_{poly.material_index}"].pass_index
        face_id_dict["-".join(vert_key)] = mat.pass_index  # mat.pass_index 作为面ID

    json_output_pth = os.path.join(output_dir, "face2id.json")
    json.dump(face_id_dict, open(json_output_pth, "w"), indent=4)
    print(f"为 {len(mesh.polygons)} 个面分配了唯一材质")

    # -------------------------
    # 渲染设置
    # -------------------------
    scene = bpy.context.scene
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'  # 启用 GPU 渲染
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()
    prefs.compute_device_type = "CUDA"  # prefs.compute_device_type = 'OPTIX'
    # scene.cycles.device = 'GPU'      # 启用 GPU 渲染
    # prefs = bpy.context.preferences.addons["cycles"].preferences
    # prefs.compute_device_type = "OPTIX" # prefs.compute_device_type = 'OPTIX'
    # # 启用所有可用的 GPU
    # prefs.get_devices()
    # for d in prefs.devices:
    #     if 'NVIDIA' in d.name:
    #         print(d.name)
    #         d.use = True
    #     else:
    #         d.use = False
    scene.cycles.samples = 1  # ID Pass 不需要采样
    scene.render.film_transparent = True  # 背景透明

    # 禁用光线追踪/抗锯齿对 ID 的影响
    scene.cycles.use_denoising = False

    # 启用 Material Index Pass
    view_layer = scene.view_layers[0]
    # view_layer = scene.view_layers["View Layer"]
    view_layer.use_pass_material_index = True

    # -------------------------
    # 设置 Compositor 节点，输出EXR
    # -------------------------
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Render Layers 节点
    rl = tree.nodes.new('CompositorNodeRLayers')

    # EXR 输出（Material Index Pass）
    file_output_index = tree.nodes.new('CompositorNodeOutputFile')
    file_output_index.label = "Index Output"
    file_output_index.base_path = output_dir
    file_output_index.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['IndexMA'], file_output_index.inputs['Image'])

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
    # 计算物体中心
    # -------------------------
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, mathutils.Vector((0, 0, 0))) / 8.0
    radius = max((v - center).length for v in bbox)
    print("物体中心:", center, "半径:", radius)

    # 相机到物体中心的合理距离（保证物体完全进入视野）
    fov = cam_data.angle
    dist = radius / math.tan(fov / 2) * 1.2  # 乘个安全系数

    # -------------------------
    # 渲染循环
    # -------------------------
    for name, pos in cam_positions.items():
        direction = mathutils.Vector(pos).normalized()
        # cam_obj.location = mathutils.Vector(pos) * 1.2
        cam_obj.location = center + direction * dist
        look_at(cam_obj, center)
        # 设置输出路径
        file_output_index.file_slots[0].path = f"{name}_faceID"

        bpy.ops.render.render(write_still=True)

    print("Face Pass (EXR) 输出完成:", output_dir)


if __name__ == '__main__':
    try:
        print(sys.argv)
        model_path = sys.argv[8]
        save_dir = sys.argv[9]
        print("======")
        render_face(model_path, save_dir)
    except Exception as e:
        print(f"Blender 内部错误: {e}")
        sys.exit(1)

