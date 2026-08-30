import os
import subprocess
from pathlib import Path
import matplotlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

matplotlib.use('Agg')  # 无界面后台渲染
import cv2
import numpy as np


def judge_bg(img_path, cur_scale_ind):
    # 读取 RGBA 图像（含透明度）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 分离 RGBA
    b, g, r, a = cv2.split(img)

    a_f = a.astype(float) / 255.0
    rate = np.mean(a_f)
    if rate < 1e-6:
        return True
    if rate < 1e-6 and cur_scale_ind == 0:
        return True

    return False


def render_axis(obj_dir, save_dir, view_ind=0, xyz="true"):
    save_png_pth = os.path.join(save_dir, f"{view_ind}_{xyz}.png")
    if os.path.exists(save_png_pth):
        os.remove(save_png_pth)
    scales = [1.5, 2, 3, 5, 10, 15]
    scale_ind = 0
    while not os.path.exists(save_png_pth) or judge_bg(save_png_pth, scale_ind - 1):
        if scale_ind == len(scales):
            break
        print("scale_ind: ", scales[scale_ind])
        subprocess.run(
            [
                os.environ.get("UNIPHYS_BLENDER", "blender"),
                "--background",
                "--python",
                str(PROJECT_ROOT / "utils/blender/blender_axis.py"),
                "--log-level",
                "0",
                "--quiet",
                "--",
                obj_dir,
                save_png_pth,
                str(view_ind),
                xyz,
                str(scales[scale_ind]),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        scale_ind += 1


def render_point(obj_dir, save_dir, view_ind=0):
    save_png_pth = os.path.join(save_dir, f"{view_ind}.png")
    if os.path.exists(save_png_pth):
        os.remove(save_png_pth)
    scales = [1.5, 2, 3, 5, 10, 15]
    scale_ind = 0
    while not os.path.exists(save_png_pth) or judge_bg(save_png_pth, scale_ind - 1):
        if scale_ind == len(scales):
            break
        print("scale_ind: ", scales[scale_ind])
        subprocess.run(
            [
                os.environ.get("UNIPHYS_BLENDER", "blender"),
                "--background",
                "--python",
                str(PROJECT_ROOT / "utils/blender/blender_point.py"),
                "--log-level",
                "0",
                "--quiet",
                "--",
                obj_dir,
                save_png_pth,
                str(view_ind),
                str(scales[scale_ind]),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        scale_ind += 1


def render_axis_main(gpt_output_dir, mov_type, entity):
    root = gpt_output_dir
    mov_dir = os.path.join(root, mov_type)
    entity_dir = os.path.join(mov_dir, entity)
    if not os.path.exists(entity_dir):
        return
    parts = os.listdir(entity_dir)
    for part in parts:
        part_dir = os.path.join(entity_dir, part)
        if not os.path.isdir(part_dir):
            continue
        # if "14" not in part:
        #     continue

        if "obj_revolute" in mov_type:
            part_dir = os.path.join(entity_dir, part, "axis")

        save_dir = os.path.join(part_dir, "imgs")
        os.makedirs(os.path.join(save_dir), exist_ok=True)

        if len(os.listdir(save_dir)) < 4:
            render_axis(part_dir, save_dir, view_ind=0, xyz="true")
            render_axis(part_dir, save_dir, view_ind=0, xyz="false")
            render_axis(part_dir, save_dir, view_ind=1, xyz="true")
            render_axis(part_dir, save_dir, view_ind=1, xyz="false")

        if "obj_revolute" in mov_type:
            part_dir = os.path.join(entity_dir, part, "pivot")
            if not os.path.isdir(part_dir):
                continue
            save_dir = os.path.join(part_dir, "imgs")
            os.makedirs(os.path.join(save_dir), exist_ok=True)
            if len(os.listdir(save_dir)) < 2:
                render_point(part_dir, save_dir, view_ind=0)
                render_point(part_dir, save_dir, view_ind=1)
