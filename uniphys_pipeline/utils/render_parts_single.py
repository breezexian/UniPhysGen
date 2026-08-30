# -*- coding: utf-8 -*-
import os
import numpy as np
import subprocess
from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def judge_bg(img_path, cur_scale_ind):
    # 读取 RGBA 图像（含透明度）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 分离 RGBA
    b, g, r, a = cv2.split(img)

    a_f = a.astype(float) / 255.0
    rate = np.mean(a_f)
    if rate < 1e-6:
        return True
    if rate < 3e-2 and cur_scale_ind == 0:
        return True

    return False


def render_parts_single_main(cur_shape_dir):
    cur_part_dir = os.path.join(cur_shape_dir, 'objs')
    leaf_part_ids = [item.split('.')[0] for item in os.listdir(cur_part_dir) if item.endswith('.obj')]

    cur_render_dir = os.path.join(cur_shape_dir, 'imgs')
    if not os.path.exists(cur_render_dir):
        os.mkdir(cur_render_dir)

    for idx in leaf_part_ids:

        if os.path.exists(os.path.join(cur_shape_dir, "glbs")):
            part_obj = os.path.join(cur_shape_dir, "glbs", str(idx) + '.glb')
        else:
            part_obj = os.path.join(cur_shape_dir, "objs", str(idx) + '.obj')
        # if not os.path.exists(part_obj):
        #     part_obj = os.path.join(cur_part_dir, str(idx) + '.obj')
        save_part_png = os.path.join(cur_render_dir, str(idx) + '_ori_single.png')
        scales = [1.5, 3, 5, 10, 15] #[2, 3, 5, 7, 9, 10, 13, 15]  [1.2, 3, 5, 10, 15]
        scale_ind = 0
        if os.path.exists(save_part_png):
            os.remove(save_part_png)
        while not os.path.exists(save_part_png) or judge_bg(save_part_png, scale_ind-1):
            if scale_ind == len(scales):
                break
            print(scales[scale_ind])
            subprocess.run(
                [
                    os.environ.get("UNIPHYS_BLENDER", "blender"),
                    "--background",
                    "--python",
                    str(PROJECT_ROOT / "utils/blender/blender_single_part.py"),
                    "--log-level",
                    "0",
                    "--quiet",
                    "--",
                    part_obj,
                    save_part_png,
                    str(scales[scale_ind]),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )
            scale_ind += 1
