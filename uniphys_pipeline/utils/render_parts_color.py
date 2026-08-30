# -*- coding: utf-8 -*-

import argparse
import random
import time
import scipy.misc as misc
import json
import os
import sys
import numpy as np
from subprocess import call
from collections import deque
from PIL import Image


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
def render_root_mesh(v, f, save_dir, color=[0.5, 0.5, 0.5]):
    tmp_dir = os.path.abspath('./tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    tmp_obj = os.path.join(tmp_dir,
                           str(time.time()).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '.obj')
    tmp_png = tmp_obj.replace('.obj', '.png')

    tmp_mtl = export_obj(tmp_obj, v, f, color=color)

    cmd = f'blender --background --python ./utils/blender/blender.py -- {tmp_obj} {tmp_png} {save_dir}'
    call(cmd, shell=True)
    # img = misc.imread(tmp_png)
    img = Image.open(tmp_png)
    img = np.array(img)
    # img = img[100:700, 100:700, :]
    img = img.astype(np.float32)

    all_white = np.ones((img.shape), dtype=np.float32) * 255

    img_alpha = img[:, :, 3] * 1.0 / 256
    all_white_alpha = 1.0 - img_alpha

    all_white[:, :, 0] *= all_white_alpha
    all_white[:, :, 1] *= all_white_alpha
    all_white[:, :, 2] *= all_white_alpha

    img[:, :, 0] *= img_alpha
    img[:, :, 1] *= img_alpha
    img[:, :, 2] *= img_alpha

    out = img[:, :, :3] + all_white[:, :, :3]

    cmd = 'rm -rf %s %s %s' % (tmp_obj, tmp_png, tmp_mtl)
    call(cmd, shell=True)

    return out, img_alpha


def render_part_mesh(part_obj, save_dir):
    tmp_dir = os.path.abspath('./tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    tmp_png = os.path.join(tmp_dir,
                           str(time.time()).replace('.', '_') + '_' + str(random.random()).replace('.', '_') + '.png')
    cmd = f'blender --background --python ./utils/blender/blender.py -- {part_obj} {tmp_png} {save_dir}'
    # cmd = f'blender --background --python ./blender.py -- {part_obj} {tmp_png}'
    call(cmd, shell=True)
    # img = misc.imread(tmp_png)
    img = Image.open(tmp_png)
    img = np.array(img)
    # img = img[100:700, 100:700, :]
    img = img.astype(np.float32)

    all_white = np.ones((img.shape), dtype=np.float32) * 255

    img_alpha = img[:, :, 3] * 1.0 / 256
    all_white_alpha = 1.0 - img_alpha

    all_white[:, :, 0] *= all_white_alpha
    all_white[:, :, 1] *= all_white_alpha
    all_white[:, :, 2] *= all_white_alpha

    img[:, :, 0] *= img_alpha
    img[:, :, 1] *= img_alpha
    img[:, :, 2] *= img_alpha

    out = img[:, :, :3] + all_white[:, :, :3]

    cmd = 'rm -rf %s' % (tmp_png)
    call(cmd, shell=True)

    return out, img_alpha


def render_parts_color_main(cur_shape_dir):
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

    root_render, _ = render_root_mesh(root_v, root_f, cur_shape_dir)

    def render():
        for idx in leaf_part_ids:
            part_obj = os.path.join(cur_part_dir, str(idx) + '.obj')
            part_v, part_f = load_obj(part_obj)
            part_render, part_alpha = render_root_mesh(part_v, part_f, cur_shape_dir, color=[0.93, 0.0, 0.0])
            # part_obj = os.path.join(cur_shape_dir, "glbs", str(idx) + '.glb')

            # part_render, part_alpha = render_part_mesh(part_obj, cur_shape_dir)
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
            out_filename = os.path.join(cur_render_dir, str(idx) + '_color.png')

            # misc.imsave(out_filename, alpha_part)
            save_img = Image.fromarray(alpha_part)
            save_img.save(out_filename)

    render()
