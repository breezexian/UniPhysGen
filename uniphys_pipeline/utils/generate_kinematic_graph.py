import json
import networkx as nx
import pickle
import matplotlib
import os
import numpy as np
import trimesh

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_mesh(pth):
    mesh_or_scene = trimesh.load(pth, process=False)
    # handle Scene or Mesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        # mesh = mesh_or_scene.dump(concatenate=True)  # 合并所有 mesh
        mesh = mesh_or_scene.to_geometry()
    else:
        mesh = mesh_or_scene
    return mesh


def build_motion_graph(json_path):
    """
    边描述 “谁跟随谁运动”。
    如果 A 部件是刚体连接到 B 部件上，并且 A 的运动依赖 B，那么就有一条从 B → A 的边。
    边可以附带额外信息，比如：
        movement_type（旋转、平移、刚体等）
        damping_coefficient
        movement_range
    """
    # 读取 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 使用 MultiDiGraph 支持多重边
    G = nx.MultiDiGraph()
    parts = data["parts"]

    # label -> name 映射
    label_to_name = {p["label"]: p["name"] for p in parts}

    # 添加节点
    for part in parts:
        G.add_node(part["label"], name=part["name"])

    # 暂存边（用于去重）
    edge_dict = {}

    for part in parts:
        for nb in part.get("neighbors", []):
            mtype = nb.get("movement_type", "")
            parent = nb.get("parent_label")
            child = nb.get("child_label")
            labels = nb.get("labels_of_movement_group", "")
            movement_range = nb.get("movement_range", "")
            damping = nb.get("damping_coefficient", "")
            label = f"{mtype}||{movement_range}||{damping}" if movement_range else mtype
            # 推断 parent / child
            if not (parent and child):
                labels_pair = sorted(list(map(int, labels.split('-'))))
                if len(labels_pair) == 2:
                    parent, child = labels_pair

            if not (parent and child):
                continue

            key = (parent, child)
            rev_key = (child, parent)
            has_rev_key = rev_key in edge_dict
            if key not in edge_dict:
                edge_dict[key] = set()

            if mtype == "D" and (len(edge_dict[key]) > 0 or (has_rev_key and len(edge_dict[rev_key]) > 0)):
                continue
            if len(edge_dict[key]) > 0 and list(edge_dict[key])[-1].split("||")[0].strip() == "D":
                edge_dict[key].pop()
            if has_rev_key and len(edge_dict[rev_key]) > 0 and list(edge_dict[rev_key])[-1].split("||")[
                0].strip() == "D":
                edge_dict[rev_key].pop()

            mtype_exist = False  # 可能会因为damping预估不一致导致label不一样，所以这里做一个判断
            for item in edge_dict[key]:
                if mtype == item.split("||")[0].strip():
                    mtype_exist = True
                    break
            if not mtype_exist:
                edge_dict[key].add(label)
                # 如果是刚体连接 D，则添加反向边
                if mtype == "D" or mtype == "A":
                    if rev_key not in edge_dict:
                        edge_dict[rev_key] = set()
                    edge_dict[rev_key].add(label)

    # 统一加入图
    print(edge_dict)

    for (parent, child), labels in edge_dict.items():
        for label in labels:
            # 提取运动类型主词
            mtype = label.split('||')[0].strip()
            if mtype in ["D", "A"]:
                range = [0., 0.]
                damping = 0.0
            else:
                range = label.split("||")[1].strip()
                range = list(map(float, range.split()[0].split("-")))
                damping = float(label.split("||")[2].strip())
            G.add_edge(parent, child, movtype=mtype, movrange=range, damping=damping)

    return G, label_to_name


def visualize_motion_graph(G, out_path='motion_graph.png'):
    pos = nx.spring_layout(G, seed=42, k=0.5)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightsteelblue')

    # 定义颜色/线型
    edge_colors = {
        "C": "red",
        "B": "blue",
        "D": "gray",
        "A": "green",
        "default": "black",
    }
    edge_styles = {
        "C": "solid",
        "B": "dashed",
        "D": "dotted",
        "A": "dotted",
        "default": "solid",
    }

    # 绘制边
    for i, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        mtype = data.get("movtype", "default")
        color = edge_colors.get(mtype, edge_colors["default"])
        style = edge_styles.get(mtype, edge_styles["default"])
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            min_target_margin=20,
            arrowsize=20,
            edgelist=[(u, v)],
            arrowstyle='->',
            width=2,
            edge_color=color,
            style=style,
            connectionstyle=f'arc3,rad={0.15 * (key + 1)}'
        )

    # 节点标签
    # labels = {n: label_to_name.get(n, str(n)) for n in G.nodes()}
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')

    # 边标签（多种运动类型合并）
    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        key_tuple = (u, v)
        text = data.get("movtype", "none")
        if key_tuple not in edge_labels:
            edge_labels[key_tuple] = [text]
        else:
            if text not in edge_labels[key_tuple]:
                edge_labels[key_tuple].append(text)

    edge_labels = {k: "\n".join(v) for k, v in edge_labels.items()}

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5,
    )

    plt.title("Motion Dependency Graph (Unique Motions per Pair)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Motion graph saved to: {out_path}")


def get_real_vol(gpt_basic_path):
    gpt_basic_info = json.load(open(gpt_basic_path, "r"))
    obj_vol = list(map(float, gpt_basic_info["dimension"].split()[0].split("*")))
    return obj_vol


def find_pure_A_nodes(G, motion_node_info):
    """
    返回所有真正的 A 类型节点：
    该节点只参与 A 类型的边，没有任何 B/C/D 等非 A 边。
    """

    for n in G.nodes():
        ok = True

        # 出边检查
        for _, v, k, data in G.out_edges(n, keys=True, data=True):
            if data.get("movtype") != "A":
                ok = False
                break

        if not ok:
            continue

        # 入边检查
        for v, _, k, data in G.in_edges(n, keys=True, data=True):
            if data.get("movtype") != "A":
                ok = False
                break

        if ok:
            if n not in motion_node_info:
                motion_node_info[n] = {"A": {}}
            else:
                motion_node_info[n]["A"] = {}


def find_motion_nodes(G, real_to_sim_rate, motion_types=("B", "C"), rigid_type="D"):
    """
    找出所有运动节点（被 B/C 边指向的节点），并沿 B/C + E 边传播带动子节点
    返回：
        motion_node_dict: {运动节点: set(被带动的子节点)}
    """
    motion_node_dict = {}
    motion_node_info = {}
    # 1. 找出所有被指向的节点，且边类型是 B 或 C
    for node in G.nodes():
        incoming_edges = G.in_edges(node, keys=True, data=True)
        for u, v, key, data in incoming_edges:
            movtype = data.get("movtype")
            movrange = data.get("movrange")
            damping = data.get("damping")
            if movtype in motion_types:
                # v 是运动节点，被指向的节点
                if v not in motion_node_dict:
                    motion_node_dict[v] = set()
                if v not in motion_node_info:
                    motion_node_info[v] = {}
                if movtype not in motion_node_info[v]:
                    motion_node_info[v][movtype] = {}
                motion_node_dict[v].add(v)  # 自身也算运动
                if movtype in ["C"]:
                    movrange = list(np.round(np.array(movrange) * np.pi / 180.0, 5))
                if movtype == "B":
                    movrange = list(np.round(np.array(movrange) * real_to_sim_rate, 5))
                motion_node_info[v][movtype]["parent"] = u
                motion_node_info[v][movtype]["range"] = movrange
                motion_node_info[v][movtype]["damping"] = damping
    find_pure_A_nodes(G, motion_node_info)
    # 2. 对每个运动节点，深度遍历其所有出边，沿 B/C 或 D 传播
    for motion_node in motion_node_dict.keys():
        visited = set()
        stack = list(G.successors(motion_node))
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                motion_node_dict[motion_node].add(n)
                # 遍历 n 的所有出边
                for _, succ, k, data in G.out_edges(n, keys=True, data=True):
                    if data.get("movtype") in motion_types or data.get("movtype") == rigid_type:
                        if succ not in visited:
                            stack.append(succ)
    for motion_node in motion_node_dict:
        motion_node_dict[motion_node] = list(motion_node_dict[motion_node])
    return motion_node_dict, motion_node_info


def get_json_from_graph(G):
    data = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes()],
        "edges": [
            {"parent": u, "child": v, **d}
            for u, v, d in G.edges(data=True)
        ]
    }

    return data


def generate_kg_main(mesh_path, gpt_output_dir, entity):
    gpt_basic_path = os.path.join(gpt_output_dir, "gpt_basic_annotation", f"{entity}.json")  # 你的 JSON 文件路径
    save_dir = os.path.join(gpt_output_dir, "kg_res", entity)
    judge_file = os.path.join(save_dir, "motion_graph.json")
    if os.path.exists(judge_file):
        return
    G, label_to_name = build_motion_graph(gpt_basic_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    visualize_motion_graph(G, out_path=os.path.join(save_dir, "kg.png"))
    real_size = get_real_vol(gpt_basic_path)
    mesh = read_mesh(mesh_path)
    bbox = mesh.bounds  # shape (2,3): [min xyz, max xyz]
    sim_size = bbox[1] - bbox[0]  # max - min
    real_to_sim_rate = max(sim_size) / max(real_size)
    motion_node_dict, kinematic_info = find_motion_nodes(G, real_to_sim_rate)
    json.dump(label_to_name, open(os.path.join(save_dir, "label2name.json"), "w"), indent=4)
    json.dump(motion_node_dict, open(os.path.join(save_dir, "motion_node_dependency.json"), "w"), indent=4)
    json.dump(kinematic_info, open(os.path.join(save_dir, "kinematic_info.json"), "w"), indent=4)
    # 保存
    with open(os.path.join(save_dir, "motion_graph.gpickle"), "wb") as f:
        pickle.dump(G, f)

    # 读取
    # with open(os.path.join(save_dir, "motion_graph.gpickle"), "rb") as f:
    #     G = pickle.load(f)
    graph_json = get_json_from_graph(G)
    json.dump(graph_json, open(os.path.join(save_dir, "motion_graph.json"), "w"), indent=4)