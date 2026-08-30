import os
import sys
import math
import json
import trimesh
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
from .MAT import MATERIAL_PRIORS

NOT_H_E_POLICY = ["rubber", "foam", "fabric", "leather", "other"]
NOT_MAT = ["other"]


class TimeoutException(Exception):
    """自定义超时异常"""
    pass


# 超时处理函数
def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


def hv_to_pa(hv):
    return hv * 9.807e6


def gpa_to_pa(gpa_value):
    return gpa_value * 1e9


def in_range(val, r):
    return r[0] <= val <= r[1]


# level 0 数值可行区间验证 Value Feasibility Check
def check_value_ranges(part):
    errors = []

    mu = part["friction"]
    E = part["young"]
    nu = part["poisson"]
    rho = part["density"]

    if not (0.0 <= mu <= 3.0):
        errors.append(f"Unreasonable friction: {mu}")

    if E <= 0 or E > 1e12:
        errors.append(f"Unreasonable Young's modulus: {E}")

    if not (-1.0 < nu < 0.5):
        errors.append(f"Invalid Poisson ratio: {nu}")

    if rho <= 0 or rho > 30000:
        errors.append(f"Unreasonable density: {rho}")

    return errors


# level 1 材质先验 Material Prior Consistency
def check_mat_pri(part):
    errors = []
    mat = part["material"]  # 简化材质大类

    pri = MATERIAL_PRIORS.get(mat)
    if mat in NOT_MAT or pri is None:
        return errors
    # Level 0: 材质先验
    for k, key_name in [("density", "density"), ("young", "young"),
                        ("poisson", "poisson"), ("friction", "friction")]:
        if k in part and not in_range(part[k], pri[key_name]):
            errors.append(f"{k} out of {mat} range: {part[k]}")
    return errors


# level 2 部件内的物理一致性 Intra-part Physical Consistency
def check_intra_part_consistency(part):
    """
    注意：
    这是 plausibility check，不是 physics law
    非常适合写成 sanity check
    :param part:
    :return:
    """
    warnings = []
    if part["material"] in NOT_H_E_POLICY:  # rubber的H/E没有意义
        return warnings
    E = part["young"]
    H = part["hardness"]

    if E > 0 and H > 0:
        ratio = H / E
        if not (1e-3 <= ratio <= 1e-1):
            warnings.append(
                f"Inconsistent hardness/Young ratio: H/E={ratio:.2e}"
            )
    return warnings


# level 3 结构一致性规则 Structural Consistency Across Parts
def check_interact_friction(parts, mu_min=0.2):
    # 规则 2：支撑面摩擦不应极低
    warnings = []
    if parts["affordance"] <= 2 or parts["graspable"]:
        mu = parts["friction"]
        if mu < mu_min:
            warnings.append(
                f"Low-friction interact surface:affordance={parts['affordance']} (mu={mu})"
            )
    return warnings


# Level 4: 质量一致性
def check_mass(parts, object_mass=None):
    warnings = []
    total_mass = 0.0
    for name, part in parts.items():
        if "volume" in part:
            mass = part["density"] * part["volume"]  # kg/m³ * m³
            total_mass += mass
            if mass <= 0:
                warnings.append(f"{name}: non-positive mass {mass:.2f} kg")
        else:
            warnings.append(f"{name}: missing volume")

    if object_mass is not None:
        rel_err = abs(total_mass - object_mass) / max(object_mass, 1e-6)
        if rel_err > 0.5:
            warnings.append(
                f"Mass inconsistency: sum(parts)={total_mass:.2f} kg vs object={object_mass:.2f} kg, rel_err={rel_err:.2f}")

    return total_mass, warnings


def check_parts(part):
    issues = {}
    issues["val_range"] = check_value_ranges(part)
    issues["mat_pri"] = check_mat_pri(part)
    issues["intra_part"] = check_intra_part_consistency(part)
    issues["interact_friction"] = check_interact_friction(part)
    return issues


def validate(parts, object_mass=None):
    all_issues = {}
    for name, part in parts.items():
        issues = check_parts(part)
        all_issues[name] = issues

    total_mass, mass_issues = check_mass(parts, object_mass)

    all_issues["__mass__"] = {"issues": mass_issues, "mass_info": {"total_mass": total_mass, "obj_mass": object_mass}}

    return all_issues


def read_mesh(pth):
    mesh_or_scene = trimesh.load(pth, process=False)
    # handle Scene or Mesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        # mesh = mesh_or_scene.dump(concatenate=True)  # 合并所有 mesh
        mesh = mesh_or_scene.to_geometry()
    else:
        mesh = mesh_or_scene
    return mesh


def basic_validation_process_main(entity, mesh_pth, res_type_dir, gpt_anno_dir, save_root):
    save_res_pth = os.path.join(save_root, f"{entity}.json")
    if os.path.exists(save_res_pth):
        return
    obj_dir = os.path.join(res_type_dir, entity, "objs")
    gpt_file_pth = os.path.join(gpt_anno_dir, f"{entity}.json")
    if not os.path.exists(gpt_file_pth):
        return
    try:
        gpt_data = json.load(open(gpt_file_pth, "r"))
        dimension = [float(i) for i in gpt_data["dimension"].split()[0].split("*")]
        mass = float(gpt_data["mass"].split()[0])
        mesh = read_mesh(mesh_pth)
        bbox = mesh.bounds  # shape (2,3): [min xyz, max xyz]
        sim_size = bbox[1] - bbox[0]  # max - min
        sim_vol = np.prod(sim_size)
        real_vol = np.prod(dimension)
        vol_sim_to_real_rate = real_vol / sim_vol
        parts = gpt_data["parts"]
        new_parts = {}
        for part in parts:
            label = part["label"]
            density = float(part["density"].split()[0]) * 1000
            young = gpa_to_pa(part["Young's Modulus (GPa)"])
            poisson = part["Poisson's Ratio"]
            friction = part["friction_coefficient"]
            hardness = hv_to_pa(part["Hardness (HV)"])
            affordance = part["priority_rank"]
            graspable = part["graspable"]
            obj_pth = os.path.join(obj_dir, f"{label}.obj")
            part_mesh = read_mesh(obj_pth)
            # part_vol = part_mesh.volume
            # 体素分辨率：相对于 AABB
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 设置超时秒数
            try:
                pitch = max(part_mesh.extents) / 128  # 128³ 级别
                vox = part_mesh.voxelized(pitch)
                part_vol = vox.filled_count * pitch ** 3
                part_real_vol = part_vol * vol_sim_to_real_rate * 1e-6
            except TimeoutException as e:
                print(e)
                json.dump({"error": "timeout"}, open(save_res_pth, "w"), indent=4)
                return
            finally:
                signal.alarm(0)

            tmp_part = {"name": part["name"], "material": part["material"].split("/")[0], "density": density,
                        "young": young, "poisson": poisson,
                        "friction": friction, "hardness": hardness, "affordance": affordance,
                        "graspable": graspable, "volume": part_real_vol}
            new_parts[label] = tmp_part
            # print(f"mass {label} {part_real_vol * density}", mass)
        all_issues = validate(new_parts, mass)
        os.makedirs(save_root, exist_ok=True)
        json.dump(all_issues, open(save_res_pth, "w"), indent=4)
    except Exception as e:
        print(e)
        print(gpt_file_pth)
        os.makedirs(save_root, exist_ok=True)
        json.dump({"error": str(e)}, open(save_res_pth, "w"), indent=4)
        return
    return
