import os
import numpy as np
import base64
import datetime
import time
from openai import OpenAI
import re
import json
from tqdm import tqdm
import matplotlib
import matplotlib.patches as patches

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import shutil


def transparent_to_white(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(img)
    white_bg = np.ones_like(a, dtype=np.uint8) * 255

    a_f = a.astype(float) / 255.0
    b = (b * a_f + white_bg * (1 - a_f)).astype(np.uint8)
    g = (g * a_f + white_bg * (1 - a_f)).astype(np.uint8)
    r = (r * a_f + white_bg * (1 - a_f)).astype(np.uint8)

    result = cv2.merge([b, g, r])

    return result


def save_gpt_input_img(images, save_pth, view_ind=0, point_img=False):
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 3 columns

    # Display original image on the left
    rgba_image = transparent_to_white(images[0])
    image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    if point_img:
        ax1.set_title(f'Candidate pivot points (camera view 0)')
    else:
        ax1.set_title(f'Candidate axes (camera view {view_ind})')
    ax1.axis('off')  # Turn off axis

    # Draw red rectangle around each image
    h, w = image.shape[:2]
    rect = patches.Rectangle((0, 0), w, h, linewidth=3, edgecolor="r", facecolor='none')
    ax1.add_patch(rect)

    # Display mask overlay on the middle
    rgba_image = transparent_to_white(images[1])
    image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)
    ax2.imshow(image)
    if point_img:
        ax2.set_title(f'Candidate pivot points (camera view 1)')
    else:
        ax2.set_title(f'XYZ Candidate axes (camera view {view_ind})')
    ax2.axis('off')

    # Draw red rectangle around each image
    h, w = image.shape[:2]
    rect = patches.Rectangle((0, 0), w, h, linewidth=3, edgecolor="r", facecolor='none')
    ax2.add_patch(rect)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)

    # Save the image to file
    plt.savefig(save_pth)

    plt.close()


def handler_response(response, save_file, save_dir):
    match = re.search(r"===BEGIN_JSON===(.*?)===END_JSON===", response, re.S)
    if match:
        json_str = match.group(1).strip()
        data = json.loads(json_str)
        json.dump(data, open(save_file, "w"), indent=4)
    else:
        with open(os.path.join(save_dir, "no_match_file.txt"), "a") as fr:
            fr.writelines(save_file + "\n")
        print("No JSON found in response", save_file)
        print("answer:", response)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_part_basic_info(parts_basic_info, part):
    if parts_basic_info is None:
        return None
    for part_info in parts_basic_info:
        if part == str(part_info["label"]):
            return part_info
    return None


def gpt_phys_revolute_annotation_main(part_seg_dir, save_res_dir, entity, gpt_config):
    gpt_save_dir = os.path.join(save_res_dir, "gpt_revolute_annotation")
    gpt_basic_pth = os.path.join(save_res_dir, "gpt_basic_annotation", f"{entity}.json")
    parts_basic_info = None
    if os.path.exists(gpt_basic_pth):
        parts_basic_info = json.load(open(gpt_basic_pth, "r"))["parts"]
    os.makedirs(os.path.join(gpt_save_dir), exist_ok=True)

    annotation_axis_dir = os.path.join(save_res_dir, "obj_revolute")

    entity_dir = os.path.join(annotation_axis_dir, entity)
    if not os.path.exists(entity_dir):
        return

    total_cost = 0
    start_time = datetime.datetime.now()
    print("开始时间:", start_time)

    use_model = gpt_config["use_model"]
    base_url = gpt_config["base_url"]
    api_key = gpt_config["api_key"]
    axis_sys_prompt_pth = gpt_config["axis_sys_prompt_pth"]

    parts = os.listdir(entity_dir)
    for part in parts:
        save_anno_file = os.path.join(gpt_save_dir, entity, f'{part}.json')
        if os.path.exists(save_anno_file):
            continue
        part_info = get_part_basic_info(parts_basic_info, part)
        basic_description = ""
        functional_description = ""
        movement_description = ""
        if part_info is not None:
            basic_description = part_info["Basic_description"]
            functional_description = part_info["Functional_description"]
            movement_description = part_info["Movement_description"]
        axis_dir = os.path.join(entity_dir, part, "axis")
        pivot_dir = os.path.join(entity_dir, part, "pivot")
        if not os.path.isdir(axis_dir):
            continue
        print(f'begin: entity->{entity}, part->{part}')
        img_save_dir = os.path.join(entity_dir, part, "gpt")
        os.makedirs(img_save_dir, exist_ok=True)
        image_list = []
        shutil.copy(os.path.join(part_seg_dir, entity, "gpt", f"{part}.png"),
                    os.path.join(img_save_dir, "1.png"))
        image_list.append(os.path.join(img_save_dir, "1.png"))
        for i in range(2):
            img_pths = [
                os.path.join(axis_dir, f"imgs/{i}_false.png"),
                os.path.join(axis_dir, f"imgs/{i}_true.png")
            ]
            save_gpt_input_img(img_pths, os.path.join(img_save_dir, f"{i + 2}.png"), view_ind=i)
            image_list.append(os.path.join(img_save_dir, f"{i + 2}.png"))

        img_pths = [
            os.path.join(pivot_dir, "imgs/0.png"),
            os.path.join(pivot_dir, "imgs/1.png")
        ]
        save_gpt_input_img(img_pths, os.path.join(img_save_dir, f"4.png"), point_img=True)
        image_list.append(os.path.join(img_save_dir, f"4.png"))

        save_anno_file = os.path.join(gpt_save_dir, entity, f'{part}.json')
        os.makedirs(os.path.dirname(save_anno_file), exist_ok=True)

        if os.path.exists(save_anno_file):
            return
        client = OpenAI(base_url=base_url, api_key=api_key)

        with open(axis_sys_prompt_pth, "r") as fr:
            sys_content = fr.read()
        system = sys_content

        axis_npy = np.load(os.path.join(axis_dir, "axis.npy"))
        axis_npy = np.round(axis_npy, 5)

        prompt = f"""You are a motion-axis and the corresponding pivot point prediction model.
Given the following inputs, determine which single candidate axis and pivot point best represent the true motion axis and pivot point of the target 3D part.

**Inputs provided:**

* Image A: Assembly visualization (three panels showing the target part highlighted in color and other parts in gray; rightmost panel shows the isolated part).
* Image B: Candidate axes view 1 (left panel: seven colored candidate axes; right panel: XYZ reference axes, which are also candidates).
* Image C: Candidate axes view 2 (same as Image B but from another diagonal perspective).
* Image D: Pivot-Point View-Displays two diagonal close-up views of the target part only (no other assembly parts). The part is rendered semi-transparent, revealing six opaque colored candidate pivot points — Red, Green, Blue, Yellow, Purple, Brown. 
* Motion type: REVOLUTE (rotational motion).
* Color–vector mapping: each color corresponds to a direction vector (e.g., `Red: (0.0, 0.0, 1.0)`).
 - Yellow: {axis_npy[0]}
 - Orange: {axis_npy[1]}
 - Purple: {axis_npy[2]}
 - Cyan: {axis_npy[3]}
 - Magenta: {axis_npy[4]}
 - Dark Gray: {axis_npy[5]}
 - Brown: {axis_npy[6]}
 - Red (Z-axis): {axis_npy[7]}
 - Green (X-axis): {axis_npy[8]}
 - Blue (Y-axis): {axis_npy[9]}
* Textual Descriptions of the Part:
 - Basic_description: {basic_description}
 - Functional_description: {functional_description}
 - Movement_description: {movement_description}

**Select exactly one axis and corresponding pivot point**, output following information in JSON:

1. part-description
2. axis-description
3. axis-color
4. axis-vector
5. motion-direction
6. pivot-point
7. justification
            """
        print(sys_content)
        print("======")
        print(prompt)
        content = [
            {
                "type": "text",
                "text": prompt,
            },
        ]

        for img_path in image_list:
            print(img_path)
            base64_image = encode_image(img_path)
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
                # timeout=300,
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
            info = f"失败：entity->{entity}; part->{part}"
            print(info, e)
            with open(os.path.join(gpt_save_dir, "failed.txt"), "a") as fr:
                fr.writelines(info + "\n")
            time.sleep(2)
    end_time = datetime.datetime.now()
    print("结束时间:", end_time)
