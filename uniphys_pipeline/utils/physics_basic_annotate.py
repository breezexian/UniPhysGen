import os
import datetime
import time
import numpy as np
import base64
from openai import OpenAI
import argparse
import logging
import re
import json
from tqdm import tqdm
import matplotlib
import matplotlib.patches as patches
import shutil

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


# B00IIFW2L4  B07B4W2X4Z
def transparent_to_white(img_path, save_path=None):
    # 读取 RGBA 图像（含透明度）
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 分离 RGBA
    b, g, r, a = cv2.split(img)

    # 创建白色背景
    white_bg = np.ones_like(a, dtype=np.uint8) * 255

    # 使用 alpha 混合前景和白底
    a_f = a.astype(float) / 255.0
    b = (b * a_f + white_bg * (1 - a_f)).astype(np.uint8)
    g = (g * a_f + white_bg * (1 - a_f)).astype(np.uint8)
    r = (r * a_f + white_bg * (1 - a_f)).astype(np.uint8)

    # 合并为 BGR
    result = cv2.merge([b, g, r])

    return result


def save_gpt_input_img(images, save_pth, ind):
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Display original image on the left
    rgba_image = cv2.imread(images[0])
    image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax1.set_title(f'part {ind} (camera view 0)')
    ax1.axis('off')  # Turn off axis

    # Draw red rectangle around each image
    h, w = image.shape[:2]
    rect = patches.Rectangle((0, 0), w, h, linewidth=3, edgecolor="r", facecolor='none')
    ax1.add_patch(rect)

    # Display mask overlay on the middle
    rgba_image = cv2.imread(images[1])
    image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(image)
    ax2.set_title(f'part {ind} (camera view 1)')
    ax2.axis('off')

    # Draw red rectangle around each image
    h, w = image.shape[:2]
    rect = patches.Rectangle((0, 0), w, h, linewidth=3, edgecolor="r", facecolor='none')
    ax2.add_patch(rect)

    # Display part image on the right

    rgba_image = transparent_to_white(images[2])
    image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)
    ax3.imshow(image)
    ax3.set_title(f'Part {ind} (single part {ind})')
    ax3.axis('off')

    # Draw red rectangle around each image
    h, w = image.shape[:2]
    rect = patches.Rectangle((0, 0), w, h, linewidth=3, edgecolor="r", facecolor='none')
    ax3.add_patch(rect)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)

    # Save the image to file
    plt.savefig(save_pth)

    plt.close()


def handler_response(response, save_anno_file, save_dir):
    match = re.search(r"===BEGIN_JSON===(.*?)===END_JSON===", response, re.S)
    if match:
        json_str = match.group(1).strip()
        data = json.loads(json_str)
        json.dump(data, open(save_anno_file, "w"), indent=4)
    else:
        with open(f"{save_dir}/no_match_file.txt", "a") as fr:
            fr.writelines(save_anno_file + "\n")
        print("No JSON found in response", save_anno_file)
        print("answer:", response)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt_phys_basic_annotation_main(part_seg_dir, save_res_dir, entity, gpt_config):
    gpt_save_dir = os.path.join(save_res_dir, "gpt_basic_annotation")

    os.makedirs(os.path.join(gpt_save_dir), exist_ok=True)

    total_cost = 0
    start_time = datetime.datetime.now()
    print("开始时间:", start_time)

    use_model = gpt_config["use_model"]
    base_url = gpt_config["base_url"]
    api_key = gpt_config["api_key"]
    basic_sys_prompt_pth = gpt_config["basic_sys_prompt_pth"]

    print(f'begin: entity->{entity}')
    save_anno_file = os.path.join(gpt_save_dir, f'{entity}.json')
    os.makedirs(os.path.dirname(save_anno_file), exist_ok=True)
    if os.path.exists(save_anno_file):
        return

    client = OpenAI(base_url=base_url,
                    api_key=api_key)

    ori_img_dir = os.path.join(part_seg_dir, entity, 'imgs')
    save_gpt_root = os.path.join(part_seg_dir, entity, "gpt")
    if os.path.exists(save_gpt_root):
        shutil.rmtree(save_gpt_root)
    os.makedirs(save_gpt_root, exist_ok=True)
    if not os.path.exists(os.path.join(part_seg_dir, entity, "objs")):
        return
    objs = [f for f in os.listdir(os.path.join(part_seg_dir, entity, "objs")) if ".obj" in f]

    image_pth_list = []
    for i in range(len(objs)):
        img1 = os.path.join(ori_img_dir, f"{i + 1}_ori_0.png")
        img2 = os.path.join(ori_img_dir, f"{i + 1}_ori_1.png")
        img3 = os.path.join(ori_img_dir, f"{i + 1}_ori_single.png")
        save_gpt_input_img([img1, img2, img3], os.path.join(save_gpt_root, f"{i + 1}.png"), i + 1)
        image_pth_list.append(os.path.join(save_gpt_root, f"{i + 1}.png"))

    num_part = len(image_pth_list)
    if num_part > 50:
        return
    with open(basic_sys_prompt_pth, "r") as fr:
        sys_content = fr.read()
    system = sys_content

    prompt = f"Analyze the {num_part} parts of a 3D object. Each image includes one part. The image order strictly defines the part labels: Image_1 → label 1, Image_2 → label 2, ..., Image_{num_part} → label {num_part}. Keep this order fixed in the output. Just output the object-level and part-level information for this object in JSON. " \
             f"When generating the JSON array parts, " \
             f"output them in ascending order of their labels (1 → {num_part})."
    part_name_json = os.path.join(part_seg_dir, entity, "part_name.json")
    if os.path.exists(part_name_json):
        part_names = json.load(open(part_name_json, "r"))
        part_names = dict(sorted(part_names.items(), key=lambda x: int(x[0])))
        object_name = part_names["0"]
        part_names = [f"label {key}: '{part_names[key]}'" for key in part_names]
        part_names_des = ", ".join(part_names[1:])
        if num_part == len(part_names) - 1:
            prompt = f"Analyze the {num_part} parts of a 3D object. The object is named {object_name} and its parts are named as follows: {part_names_des}. " \
                     f"Each image includes one part. The image order strictly defines the part labels: Image_1 → label 1, Image_2 → label 2, ..., Image_{num_part} → label {num_part}. Keep this order fixed in the output. Just output the object-level and part-level information for this object in JSON. " \
                     f"When generating the JSON array parts, " \
                     f"output them in ascending order of their labels (1 → {num_part})."
    print(sys_content)
    print("======")
    print(prompt)
    content = [
        {
            "type": "text",
            "text": prompt,
        },
    ]

    for img_pth in image_pth_list:
        print(img_pth)
        base64_image = encode_image(img_pth)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        )
    try:
        response = client.chat.completions.create(
            model=use_model,
            reasoning_effort="medium",
            #timeout=300,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            # temperature=0,
        )

        handler_response(response.choices[0].message.content, save_anno_file, gpt_save_dir)
        expense = 10 * response.usage.completion_tokens / 1000000 + 1.25 * response.usage.prompt_tokens / 1000000
        total_cost += expense
        print('cur_cost: ' + str(expense))
        print('total_cost: ' + str(total_cost))
    except Exception as e:
        info = f"失败：entity->{entity}"
        print(info, e)
        with open(os.path.join(gpt_save_dir, "failed.txt"), "a") as fr:
            fr.writelines(info + "\n")
        time.sleep(2)
    end_time = datetime.datetime.now()
    print("结束时间:", end_time)