# -*- coding: utf-8 -*-

import random
import time
import os
import numpy as np
from PIL import Image
import cv2
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f


def export_obj(out, v, f, color):
    mtl_out = out.replace('.obj', '.mtl')

    with open(out, 'w') as fout:
        fout.write('mtllib %s\n' % os.path.basename(mtl_out))
        fout.write('usemtl m1\n')
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    with open(mtl_out, 'w') as fout:
        fout.write('newmtl m1\n')
        fout.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout.write('Ka 0 0 0\n')

    return mtl_out


# [0.216, 0.494, 0.722]
def render_root_mesh(v, f, save_dir, view_index, color=[0.5, 0.5, 0.5]):
    tmp_name = hashlib.md5(save_dir.encode('utf-8')).hexdigest()
    tmp_dir = tempfile.mkdtemp(prefix=f'uniphys_{tmp_name}_')

    tmp_obj = os.path.join(tmp_dir,
                           str(time.time()).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '.obj')
    tmp_png = tmp_obj.replace('.obj', '.png')

    export_obj(tmp_obj, v, f, color=color)

    try:
        subprocess.run(
            [
                os.environ.get("UNIPHYS_BLENDER", "blender"),
                "--background",
                "--python",
                str(PROJECT_ROOT / "utils/blender/blender.py"),
                "--log-level",
                "0",
                "--quiet",
                "--",
                tmp_obj,
                tmp_png,
                save_dir,
                str(view_index),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        img = np.array(Image.open(tmp_png)).astype(np.float32)
        all_white = np.ones((img.shape), dtype=np.float32) * 255
        img_alpha = img[:, :, 3] * 1.0 / 256
        all_white_alpha = 1.0 - img_alpha
        all_white[:, :, 0] *= all_white_alpha
        all_white[:, :, 1] *= all_white_alpha
        all_white[:, :, 2] *= all_white_alpha
        img[:, :, 0] *= img_alpha
        img[:, :, 1] *= img_alpha
        img[:, :, 2] *= img_alpha
        return img[:, :, :3] + all_white[:, :, :3]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_part_mesh(part_obj, save_dir, view_index):
    tmp_name = hashlib.md5(save_dir.encode('utf-8')).hexdigest()
    tmp_dir = tempfile.mkdtemp(prefix=f'uniphys_{tmp_name}_')

    tmp_png = os.path.join(tmp_dir,
                           str(time.time()).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '.png')
    try:
        subprocess.run(
            [
                os.environ.get("UNIPHYS_BLENDER", "blender"),
                "--background",
                "--python",
                str(PROJECT_ROOT / "utils/blender/blender.py"),
                "--log-level",
                "0",
                "--quiet",
                "--",
                part_obj,
                tmp_png,
                save_dir,
                str(view_index),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        img = np.array(Image.open(tmp_png)).astype(np.float32)
        all_white = np.ones((img.shape), dtype=np.float32) * 255
        img_alpha = img[:, :, 3] * 1.0 / 256
        all_white_alpha = 1.0 - img_alpha
        all_white[:, :, 0] *= all_white_alpha
        all_white[:, :, 1] *= all_white_alpha
        all_white[:, :, 2] *= all_white_alpha
        img[:, :, 0] *= img_alpha
        img[:, :, 1] *= img_alpha
        img[:, :, 2] *= img_alpha
        return img[:, :, :3] + all_white[:, :, :3], img_alpha
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def judge_bg(img_path):
    # 读取 RGBA 图像（含透明度）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 分离 RGBA
    b, g, r, a = cv2.split(img)

    a_f = a.astype(float) / 255.0

    if np.mean(a_f) < 1e-6:
        return True

    return False


def render_parts_ori_main(cur_shape_dir, view_index):
    cur_part_dir = os.path.join(cur_shape_dir, 'objs')
    leaf_part_ids = [item.split('.')[0] for item in os.listdir(cur_part_dir) if item.endswith('.obj')]

    cur_render_dir = os.path.join(cur_shape_dir, 'imgs')
    if not os.path.exists(cur_render_dir):
        os.mkdir(cur_render_dir)

    root_v_list = []
    root_f_list = []
    tot_v_num = 0
    for idx in leaf_part_ids:
        v, f = load_obj(os.path.join(cur_part_dir, str(idx) + '.obj'))
        mesh = dict()
        mesh['v'] = v
        mesh['f'] = f
        root_v_list.append(v)
        root_f_list.append(f + tot_v_num)
        tot_v_num += v.shape[0]

    root_v = np.vstack(root_v_list)
    root_f = np.vstack(root_f_list)

    root_render = render_root_mesh(root_v, root_f, cur_shape_dir, view_index)

    def render():
        for idx in leaf_part_ids:
            # part_obj = os.path.join(cur_part_dir, str(idx) + '.obj')
            if os.path.exists(os.path.join(cur_shape_dir, "glbs")):
                part_obj = os.path.join(cur_shape_dir, "glbs", str(idx) + '.glb')
            else:
                part_obj = os.path.join(cur_shape_dir, "objs", str(idx) + '.obj')

            part_render, part_alpha = render_part_mesh(part_obj, cur_shape_dir, view_index)

            root_alpha = np.ones(root_render.shape[0:2])
            root_alpha[part_alpha > 0.5] = 0.2
            root_alpha[part_alpha <= 0.5] = 0.3
            part_alpha[part_alpha > 0.5] = 0.8
            part_alpha[part_alpha <= 0.5] = 0.7
            root_alpha = root_alpha[:, :, None]
            part_alpha = part_alpha[:, :, None]

            # alpha_part = 0.3 * root_render + 0.7 * part_render
            alpha_part = root_alpha * root_render + part_alpha * part_render
            alpha_part = alpha_part.astype(np.uint8)
            out_filename = os.path.join(cur_render_dir, str(idx) + f'_ori_{view_index}.png')

            # misc.imsave(out_filename, alpha_part)
            save_img = Image.fromarray(alpha_part)
            save_img.save(out_filename)

    render()

