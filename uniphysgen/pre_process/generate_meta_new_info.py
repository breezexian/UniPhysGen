import os
import json
import shutil
from tqdm import tqdm
import numpy as np

def get_nodes(entity_dir):
    nodes = set()
    obj_dir = os.path.join(entity_dir, "objs")
    for file in os.listdir(obj_dir):
        if file.endswith(".obj"):
            nodes.add(int(file.split(".")[0]))
    return nodes


if __name__ == '__main__':
    bench_root = "/data-koolab-nas/xianzi/uniphys-40k_new"
    cnt = 0
    damp_error = 0
    not_eqaul = 0
    ens = []
    selected = []
    selected = json.load(open("/data-koolab-nas/xianzi/code/uniphysgen/pre_process/final_uniphys_40k_sampled.json", "r"))["extra"]
    cnt = 39919
    print(len(selected))
    for ds_name in os.listdir(bench_root):
        ds_dir = os.path.join(bench_root, ds_name)
        for type_name in tqdm(os.listdir(ds_dir)):
            type_dir = os.path.join(ds_dir, type_name)
            # cnt += len(os.listdir(type_dir))
            # continue
            for entity in os.listdir(type_dir):
                if len(selected) > 0 and entity not in selected:
                    continue
                entity_dir = os.path.join(type_dir, entity)
                obj_dir = os.path.join(entity_dir, "parts")
                anno_dir = os.path.join(entity_dir, "annotations")
                if not os.path.exists(anno_dir):
                    assert 1==2
                meta_pth = os.path.join(entity_dir, "meta_data.json")
                meta_info = json.load(open(meta_pth, "r"))

                new_entity = f"UP_{str(cnt).zfill(8)}"
                ens.append(new_entity)
                meta_info["id"] = new_entity
                new_pth = os.path.join(entity_dir, "meta_data_new.json")
                if os.path.exists(new_pth):
                    assert 1==2
                json.dump(meta_info, open(new_pth, "w"), indent=4)
                cnt += 1
    print(cnt)
    print(len(ens), len(set(ens)))
    print(max(ens))
    print(sorted(ens))