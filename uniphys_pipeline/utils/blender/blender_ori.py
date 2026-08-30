import bpy
import os
import mathutils
import math
import sys


def rotate_all_mesh_geometry_x_minus_90():
    R = mathutils.Matrix.Rotation(-math.pi / 2, 4, 'X')

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.transform(R)
            obj.data.update()


def compute_scene_bbox_center_radius(mesh_objs):
    all_corners = []

    for obj in mesh_objs:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ mathutils.Vector(corner))

    all_corners = list(all_corners)

    center = sum(all_corners, mathutils.Vector((0, 0, 0))) / len(all_corners)
    radius = max((v - center).length for v in all_corners)

    return center, radius


def render_ori(glb_path, output_dir):
    # -------------------------
    # 参数设置
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)

    render_width = 800
    render_height = 800

    # 八个对角方向
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

    print("==== Scene Objects ====")
    for o in bpy.context.scene.objects:
        print(o.name, o.type, "parent:", o.parent.name if o.parent else None,
              "rot:", o.rotation_euler)

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
    dist = radius / math.tan(fov / 2) * 1.2  # 乘个安全系数

    # # -------------------------
    # # 添加点光源
    # # -------------------------
    # light_data = bpy.data.lights.new(name="KeyLight", type='POINT')
    # light_data.energy = 1000  # 可调节亮度
    # light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    # scene.collection.objects.link(light_obj)
    # light_obj.location = center + mathutils.Vector((2 * radius, 2 * radius, 2 * radius))

    # -------------------------
    # 渲染循环
    # -------------------------
    for name, pos in cam_positions.items():
        # 设置相机位置（保持和 Face ID 渲染一致）
        direction = mathutils.Vector(pos).normalized()
        # cam_obj.location = mathutils.Vector(pos) * 1.2
        cam_obj.location = center + direction * dist
        # look_at(cam_obj, mathutils.Vector((0, 0, 0)))
        look_at(cam_obj, center)
        # 输出路径
        scene.render.filepath = os.path.join(output_dir, f"{name}.png")
        # 渲染
        bpy.ops.render.render(write_still=True)

    print("正常材质渲染完成，保存在:", output_dir)


if __name__ == '__main__':

    try:
        print(sys.argv)
        model_path = sys.argv[8]
        save_dir = sys.argv[9]
        print(model_path, save_dir)
        render_ori(model_path, save_dir)
    except Exception as e:
        print(f"Blender 内部错误: {e}")
        sys.exit(1)
