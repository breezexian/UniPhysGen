import json

pth = "/seaweedfs/xianzi/code/physics_annotation_pipeline/mats_sub2.json"
dt = json.load(open(pth, "r"))
words = set()
for key in dt:
    tmp = key.split("_")
    for w in tmp:
        words.add(w)

print(len(words))
print(",".join(list(words)))