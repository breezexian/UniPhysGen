import bpy
import os
import mathutils
import math
import sys
import numpy as np

print(sys.argv)
model_path = sys.argv[8]
png_path = sys.argv[9]
dist_scale = float(sys.argv[10])


render_width = 600
render_height = 600


cam_positions = [[0.5, -1, 0.5], [0.5, 1, 0.5]]
pos = cam_positions[0]

# -------------------------
# 清空场景
# -------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# -------------------------
# 导入模型
# -------------------------
if model_path.endswith(".glb") or model_path.endswith(".gltf"):
    bpy.ops.import_scene.gltf(filepath=model_path)
elif model_path.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=model_path)
    # bpy.ops.import_scene.obj(filepath=model_path)
else:
    raise ValueError("不支持的文件格式: " + model_path)

# 找到第一个 Mesh 对象
mesh_objs = [o for o in bpy.context.scene.collection.objects if o.type == 'MESH']
if len(mesh_objs) == 0:
    raise ValueError("没有找到 Mesh 对象，请检查 glb 文件！")
obj = mesh_objs[0]

print("找到 Mesh 对象:", obj.name)

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
scene.render.film_transparent = True  # 背景透明，可改成 False

# 创建或获取 World
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)  # 白色
bg.inputs[1].default_value = 1.0  # 强度

# 创建相机
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
dist = radius / math.tan(fov / 2) * dist_scale  # 乘个安全系数

# -------------------------
# 渲染
# -------------------------
direction = mathutils.Vector(pos).normalized()
cam_obj.location = center + direction * dist
look_at(cam_obj, center)
# 输出路径
scene.render.filepath = png_path
# 渲染
bpy.ops.render.render(write_still=True)
