import json
import networkx as nx
import matplotlib
import os
import numpy as np
import trimesh
import re
from subprocess import call, check_call

matplotlib.use('Agg')
import matplotlib.pyplot as plt

pivot_color_dict = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "purple": 4,
    "brown": 5
}

axis_color_dict = {
    "yellow": 0,
    "orange": 1,
    "purple": 2,
    "cyan": 3,
    "magenta": 4,
    "dark gray": 5,
    "brown": 6,
    "red": 7,
    "green": 8,
    "blue": 9,
}


def parse_axis_vector(s):
    """
    从任意 LLM 风格字符串中解析 3D 向量
    支持:
    - [0.0e+00 2.3e-04 1.0e+00]
    - [0.0, 2.3e-04, 1.0]
    - 0.0, 2.3e-04, 1.0
    """
    # 1. 如果已经是numpy数组，直接返回
    if isinstance(s, (list, tuple)):
        return s
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) != 3:
        raise ValueError(f"Invalid axis vector: {s}")
    return np.array([float(x) for x in nums])


# a = "[ 9.8439e-01 -1.7602e-01  4.6000e-04]"
# print(parse_axis_vector(a))
# assert 1==2

def generate_revolute_mujoco_xml(motion_node_dict, kinematic_info, entity, all_nodes, part_obj_dir, gpt_dir,
                                 save_res_dir):
    # init_pos_z = 0
    # if "partnet" in gpt_dir:
    #     init_pos_z = 0.8
    revolute_dir = os.path.join(gpt_dir, "obj_revolute", entity)
    gpt_revolute_dir = os.path.join(gpt_dir, "gpt_revolute_annotation", entity)
    if not os.path.exists(revolute_dir) or not os.path.exists(gpt_revolute_dir):
        return
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    print(save_res_dir)
    part_obj_dir = os.path.abspath(part_obj_dir)
    revolute_parts = os.listdir(revolute_dir)
    for motion_node in motion_node_dict:
        if motion_node not in revolute_parts or motion_node not in all_nodes:
            continue
        gpt_anno_res = json.load(open(os.path.join(gpt_revolute_dir, f"{motion_node}.json")))
        candidate_axis_color = gpt_anno_res["axis-color"].lower().strip().split()[0]
        try:
            candidate_axis_vector = np.array(list(eval(gpt_anno_res["axis-vector"])))
        except Exception as e:
            print(e)
            print(gpt_anno_res["axis-vector"])
            print(revolute_dir, "lilili")
            candidate_axis_vector = parse_axis_vector(gpt_anno_res["axis-vector"])
        pivot_point_color = gpt_anno_res["pivot-point"].lower().strip().split()[0]
        pivot_points = np.load(os.path.join(revolute_dir, f"{motion_node}/pivot/pivot.npy"))
        axis_vectors = np.load(os.path.join(revolute_dir, f"{motion_node}/axis/axis.npy"))
        axis_vectors = np.round(axis_vectors, 5)
        try:
            axis_vector = axis_vectors[axis_color_dict[candidate_axis_color]]

            assert np.array_equal(candidate_axis_vector,
                                  axis_vector), f"the gpt selected vector {candidate_axis_vector} is not equal to the " \
                                                f"color selected counterpart {axis_vector} "
            if pivot_point_color == "average":
                pivot_point = np.mean(pivot_points, axis=0)
            else:
                pivot_point = pivot_points[pivot_color_dict[pivot_point_color]]
        except Exception as e:
            print("revolute error", e)
            continue
        motion_range = kinematic_info[motion_node]["C"]["range"]
        # if gpt_anno_res["motion-direction"] == "negative":
        #     motion_range = np.array(motion_range) * -1
        damping = kinematic_info[motion_node]["C"]["damping"]
        tmp_child_nodes = set(list(map(str, motion_node_dict[motion_node])))
        child_nodes = set()
        for node in tmp_child_nodes:
            if node not in all_nodes:
                continue
            child_nodes.add(node)
        base_nodes = set(all_nodes) - child_nodes
        child_nodes -= {motion_node}
        asset_str = ""
        base_str = ""
        child_str = ""
        for node in all_nodes:
            node_file = f"{node}.obj"
            asset_str += f'\t\t<mesh name="mesh_{node}" file="{node_file}"/>\n'
        for node in base_nodes:
            base_str += f'\t\t\t<geom type="mesh" mesh="mesh_{node}" rgba="0.7 0.7 0.7 0.1"/>\n'
        for node in child_nodes:
            child_str += f"""
                <body name="part_{node}">
                    <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
                    <geom type="mesh" mesh="mesh_{node}" rgba="0.7 0.7 0.7 0.1"/>
                </body>
            """

        mujoco_xml = f"""<mujoco model="mesh_back_office_chair">
    <visual>
        <global offwidth="1080" offheight="1080"/>
    </visual>

    <compiler angle="radian" coordinate="local" meshdir="{part_obj_dir}"/>
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <default>
        <geom type="mesh" contype="1" conaffinity="1" condim="3"
              friction="0.8 0.1 0.1"/>
    </default>

    <asset>
{asset_str}
    </asset>

    <worldbody>
        <geom name="floor"
              type="plane"
              pos="0 0 -2"
              size="200 200 0.1"
              rgba="3 3 3 1"
              friction="1 0.005 0.0001"/>
        <body name="part_base" pos="0 0 0" euler="1.5708 0 0">
            <inertial pos="0 0 0" mass="3" diaginertia="0.2 0.2 0.2"/>
{base_str}

            <body name="motion_part">
                <inertial pos="0 0 0" mass="5" diaginertia="0.2 0.2 0.2"/>
                <geom name="part_visual" type="mesh" mesh="mesh_{motion_node}" rgba="2.5 0 0 1" contype="0"
                      conaffinity="0"/>

                <joint name="joint_1" type="hinge"
                       pos="{pivot_point[0]} {pivot_point[1]} {pivot_point[2]}"
                       axis="{axis_vector[0]} {axis_vector[1]} {axis_vector[2]}"
                       limited="false"
                       range="{motion_range[0]} {motion_range[1]}"
                       damping="{damping}"/>
                {child_str}
            </body>
        </body>
    </worldbody>
</mujoco>
        """
        xml_path = os.path.join(save_res_dir, f"{motion_node}_revolute.xml")
        with open(xml_path, "w") as f:
            f.write(mujoco_xml)


def generate_prismatic_mujoco_xml(motion_node_dict, kinematic_info, entity, all_nodes, part_obj_dir, gpt_dir,
                                  save_res_dir):
    # init_pos_z = 0
    # if "partnet" in gpt_dir:
    #     init_pos_z = 0.8
    prismatic_dir = os.path.join(gpt_dir, "obj_prismatic", entity)
    gpt_prismatic_dir = os.path.join(gpt_dir, "gpt_prismatic_annotation", entity)
    if not os.path.exists(prismatic_dir) or not os.path.exists(gpt_prismatic_dir):
        return
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir)
    part_obj_dir = os.path.abspath(part_obj_dir)
    prismatic_parts = os.listdir(prismatic_dir)
    for motion_node in motion_node_dict:
        if motion_node not in prismatic_parts or motion_node not in all_nodes:
            continue
        gpt_anno_res = json.load(open(os.path.join(gpt_prismatic_dir, f"{motion_node}.json")))
        candidate_axis_color = gpt_anno_res["axis-color"].lower().strip().split()[0]
        try:
            candidate_axis_vector = np.array(list(eval(gpt_anno_res["axis-vector"])))
        except Exception as e:
            print(e)
            candidate_axis_vector = parse_axis_vector(gpt_anno_res["axis-vector"])
        axis_vectors = np.load(os.path.join(prismatic_dir, f"{motion_node}/axis.npy"))
        axis_vectors = np.round(axis_vectors, 5)
        try:
            axis_vector = axis_vectors[axis_color_dict[candidate_axis_color]]
            assert np.array_equal(candidate_axis_vector,
                                  axis_vector), f"the gpt selected vector {candidate_axis_vector} is not equal to the " \
                                                f"color selected counterpart {axis_vector} "
        except Exception as e:
            print("prismatic error", e)
            continue
        motion_range = kinematic_info[motion_node]["B"]["range"]
        # if gpt_anno_res["motion-direction"] == "negative":
        #     motion_range = np.array(motion_range) * -1
        damping = kinematic_info[motion_node]["B"]["damping"]
        tmp_child_nodes = set(list(map(str, motion_node_dict[motion_node])))
        child_nodes = set()
        for node in tmp_child_nodes:
            if node not in all_nodes:
                continue
            child_nodes.add(node)
        base_nodes = set(all_nodes) - child_nodes
        child_nodes -= {motion_node}
        asset_str = ""
        base_str = ""
        child_str = ""
        for node in all_nodes:
            node_file = f"{node}.obj"
            asset_str += f'\t\t<mesh name="mesh_{node}" file="{node_file}"/>\n'
        for node in base_nodes:
            base_str += f'\t\t\t<geom type="mesh" mesh="mesh_{node}" rgba="0.7 0.7 0.7 0.1"/>\n'
        for node in child_nodes:
            child_str += f"""
                <body name="part_{node}">
                    <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
                    <geom type="mesh" mesh="mesh_{node}" rgba="0.7 0.7 0.7 0.1"/>
                </body>
            """

        mujoco_xml = f"""<mujoco model="mesh_back_office_chair">
    <visual>
        <global offwidth="1080" offheight="1080"/>
    </visual>

    <compiler angle="radian" coordinate="local" meshdir="{part_obj_dir}"/>
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <default>
        <geom type="mesh" contype="1" conaffinity="1" condim="3"
              friction="0.8 0.1 0.1"/>
    </default>

    <asset>
{asset_str}
    </asset>

    <worldbody>
        <geom name="floor"
              type="plane"
              pos="0 0 -2"
              size="200 200 0.1"
              rgba="3 3 3 1"
              friction="1 0.005 0.0001"/>
        <body name="part_base" pos="0 0 0" euler="1.5708 0 0">
            <inertial pos="0 0 0" mass="3" diaginertia="0.2 0.2 0.2"/>
{base_str}

            <body name="motion_part">
                <inertial pos="0 0 0" mass="5" diaginertia="0.2 0.2 0.2"/>
                <geom name="part_visual" type="mesh" mesh="mesh_{motion_node}" rgba="2.5 0 0 1" contype="0"
                      conaffinity="0"/>

                <joint name="joint_1" type="slide"
                       pos="0 0 0"
                       axis="{axis_vector[0]} {axis_vector[1]} {axis_vector[2]}"
                       limited="false"
                       range="{motion_range[0]} {motion_range[1]}"
                       damping="{damping}"/>
                {child_str}
            </body>
        </body>
    </worldbody>
</mujoco>
        """
        xml_path = os.path.join(save_res_dir, f"{motion_node}_prismatic.xml")
        with open(xml_path, "w") as f:
            f.write(mujoco_xml)


def get_useful_objs(res_type_dir, entity):
    obj_dir = os.path.join(res_type_dir, entity, "merge_objs")
    if not os.path.exists(obj_dir):
        obj_dir = os.path.join(res_type_dir, entity, "objs")
    objs = []
    files = os.listdir(obj_dir)
    for file in files:
        if not file.endswith(".obj"):
            continue
        label = file.split(".")[0]
        objs.append(label)
    return obj_dir, objs


def generate_kg_mujoco_xml(part_seg_dir, gpt_output_dir, entity):
    kg_res_dir = os.path.join(gpt_output_dir, "kg_res", entity)
    if not os.path.exists(kg_res_dir):
        return
    save_xml_dir = os.path.join(gpt_output_dir, "kg_xml", entity)
    if os.path.exists(save_xml_dir):
        xml_files = list(os.listdir(save_xml_dir))
        has_video = False
        for xml_file in xml_files:
            if xml_file.endswith(".mp4"):
                has_video = True
                break
        cur_xml_nums = len(xml_files)
        # if not has_video:
        #     cmd = f"rm -rf {save_xml_dir}"
        #     call(cmd, shell=True)
        if has_video and cur_xml_nums > 0:
            revolute_dir = os.path.join(gpt_output_dir, "obj_revolute", entity)
            prismatic_dir = os.path.join(gpt_output_dir, "obj_prismatic", entity)
            expected_xml_nums = 0
            if os.path.exists(revolute_dir):
                expected_xml_nums += len(list(os.listdir(revolute_dir)))
            if os.path.exists(prismatic_dir):
                expected_xml_nums += len(list(os.listdir(prismatic_dir)))
            if cur_xml_nums >= expected_xml_nums:
                print(entity, "completed!")
                return

    label_to_name = json.load(open(os.path.join(kg_res_dir, "label2name.json"), "r"))
    motion_node_dict = json.load(open(os.path.join(kg_res_dir, "motion_node_dependency.json"), "r"))
    kinematic_info = json.load(open(os.path.join(kg_res_dir, "kinematic_info.json"), "r"))

    #obj_dir = os.path.join(part_seg_dir, f"{entity}/objs")
    #all_parts = list(label_to_name.keys())
    obj_dir, all_parts = get_useful_objs(part_seg_dir, entity)
    #all_parts = list(set(all_parts) - {"18"})

    generate_revolute_mujoco_xml(motion_node_dict, kinematic_info, entity, all_parts, obj_dir, gpt_output_dir,
                                 save_res_dir=save_xml_dir)
    generate_prismatic_mujoco_xml(motion_node_dict, kinematic_info, entity, all_parts, obj_dir, gpt_output_dir,
                                  save_res_dir=save_xml_dir)

