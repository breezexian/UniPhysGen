import bpy
import os
import mathutils
import math
import sys
import numpy as np
import json

# ============ 路径设置 ============
model_dir = sys.argv[8]
png_path = sys.argv[9]
view_index = int(sys.argv[10])
dist_scale = float(sys.argv[11])

color_list = [
    (1.0, 0.0, 0.0, 1.0),  # 红
    (0.0, 1.0, 0.0, 1.0),  # 绿
    (0.0, 0.0, 1.0, 1.0),  # 蓝
    (1.0, 1.0, 0.0, 1.0),  # 黄
    (0.6, 0.2, 1.0, 1.0),  # 紫
    (0.4, 0.25, 0.1, 1.0),  # Brown
    (0.0, 1.0, 1.0, 1.0),  # 青
    (1.0, 0.5, 0.0, 1.0),  # 橙
    (1.0, 0.0, 1.0, 1.0),  # 品红
    (0.275, 0.275, 0.275, 1.0),  # 深灰
    (0.4, 0.25, 0.1, 1.0),  # Brown

]
ids = [1, 2, 3, 4, 5]

# ============ 渲染参数 ============
render_width = 800
render_height = 800
cam_positions = [[0.5, -1, 0.5], [0.5, 1, 0.5]]
pos = cam_positions[view_index]

# -------------------------
# 清空场景
# -------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ============ 批量导入模型 ============
imported_objs = {}  # 存储已经导入的对象
for file in os.listdir(model_dir):
    if file.endswith(".obj"):
        if "ori" in file:
            continue
        name_in_blender = os.path.splitext(file)[0]  # 去掉后缀，作为Blender中的名称

        # 检查对象是否已经导入，如果没有，则导入
        if name_in_blender not in imported_objs:
            path = os.path.join(model_dir, file)
            # bpy.ops.import_scene.obj(filepath=path)
            bpy.ops.wm.obj_import(filepath=path)
            imported_objs[name_in_blender] = bpy.data.objects[name_in_blender]
            print("导入：", file)
        else:
            print(f"已导入：{file}, 跳过")
# 获取最近导入的对象
# imported_objs = bpy.context.selected_objects
# for obj in imported_objs:
#     print(obj.name)  # 实际 Blender 名称
# mesh_objs = [o for o in bpy.context.scene.collection.objects if o.type == 'MESH']
# for mesh in mesh_objs:
#     print(mesh.name)
# 如果没有任何导入的对象，抛出异常
if len(imported_objs) == 0:
    raise ValueError("未导入任何 OBJ 文件！")

# 将导入的对象放入 objs 列表
objs = list(imported_objs.values())

# 识别主模型（非 axis 对象）
main_objs = [o for o in objs if "point" not in o.name.lower()]
axis_names = [o.name for o in objs if "point" in o.name.lower()]
axis_names = sorted(axis_names, key=lambda x: int(x.split('_')[1]))
axis_objs = []
for axis_name in axis_names:
    axis_objs.append(imported_objs[axis_name])
# axis_objs = [o for o in objs if "axis" in o.name.lower()]
for o in axis_objs:
    print(f"{o.name.lower()}")

if len(main_objs) == 0:
    main_obj = objs[0]
else:
    main_obj = main_objs[0]

print(f"主模型: {main_obj.name},  轴对象数: {len(axis_objs)}")

# ============ 主模型透明化 ============
transparent_mat = bpy.data.materials.new(name="MainTransparent")
transparent_mat.use_nodes = True

# 获取默认的 Principled BSDF 节点
bsdf = transparent_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # 灰白色，可改
bsdf.inputs['Roughness'].default_value = 0.4
bsdf.inputs['Alpha'].default_value = 0.2  # 透明度：1=不透明, 0=完全透明

# 启用混合模式
transparent_mat.blend_method = 'BLEND'
# transparent_mat.shadow_method = 'HASHED'
transparent_mat.use_backface_culling = False

# 将材质赋予主模型
main_obj.data.materials.clear()
main_obj.data.materials.append(transparent_mat)

print(f"主模型 {main_obj.name} 设置为透明材质")

# ============ 渲染设置 ============
scene = bpy.context.scene
scene.render.resolution_x = render_width
scene.render.resolution_y = render_height
scene.render.engine = 'CYCLES'
# scene.cycles.device = 'GPU'  # 启用 GPU 渲染
# prefs = bpy.context.preferences.addons["cycles"].preferences
# prefs.get_devices()
# prefs.compute_device_type = "CUDA"  # prefs.compute_device_type = 'OPTIX'

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
scene.cycles.samples = 128
scene.render.film_transparent = True

if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 1.0

# ============ 创建相机 ============
cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj


def look_at(cam, target):
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


# ============ 计算所有 OBJ 的整体中心 ============
all_points = []
for o in objs:
    # 每个对象的 8 个包围盒顶点坐标（转到世界坐标）
    bbox = [o.matrix_world @ mathutils.Vector(corner) for corner in o.bound_box]
    all_points.extend(bbox)

# 计算整体包围盒
min_corner = mathutils.Vector((
    min(p.x for p in all_points),
    min(p.y for p in all_points),
    min(p.z for p in all_points),
))
max_corner = mathutils.Vector((
    max(p.x for p in all_points),
    max(p.y for p in all_points),
    max(p.z for p in all_points),
))

center = (min_corner + max_corner) / 2
radius = (max_corner - center).length  # 半径 = 包围盒对角线一半

fov = cam_data.angle
dist = radius / math.tan(fov / 2) * dist_scale

# ============ 相机定位 ============
direction = mathutils.Vector(pos).normalized()
cam_obj.location = center + direction * dist
look_at(cam_obj, center)

for i, ax in enumerate(axis_objs):
    color = color_list[i % len(color_list)]
    mat = bpy.data.materials.new(name=f"axis_color_{i}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.4
    ax.data.materials.clear()
    ax.data.materials.append(mat)
    print(f"为 {ax.name} 赋色 {color}")

# ============ 渲染输出 ============
scene.render.filepath = png_path
bpy.ops.render.render(write_still=True)
print("渲染完成:", png_path)
