import numpy as np
import os

cls = set()
root = "/home/xianzi/code/PartField/exp_results/clustering/abo/cluster_out/"
for file in os.listdir(root):
    label = np.load(os.path.join(root, file))
    for i in list(np.unique(label)):
        cls.add(i)
    print(file, label.shape, len(np.unique(label)))
print(cls)
print(len(cls))
