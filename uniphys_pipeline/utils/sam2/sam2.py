import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
from contextlib import nullcontext

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.sam_utils import create

# 如果可用，启用 CUDA
autocast_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if torch.cuda.is_available()
    else nullcontext()
)
autocast_context.__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 设置设备为 CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Generate SAM2 masks for rendered views.")
parser.add_argument("--entity_dir", type=str, required=True, help="Rendered entity directory")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./checkpoints/sam2_hiera_large.pt",
    help="SAM2 model checkpoint",
)
parser.add_argument(
    "--model-config",
    type=str,
    default="sam2_hiera_l.yaml",
    help="SAM2 Hydra model configuration",
)
args = parser.parse_args()

# 构建 SAM 2 模型
sam2_model = build_sam2(
    args.model_config,
    args.checkpoint,
    device=DEVICE,
    apply_postprocessing=False,
)

# 创建自动掩码生成器
mask_generator = SAM2AutomaticMaskGenerator(sam2_model,
                                            points_per_side=32,
                                            pred_iou_thresh=0.7, #0.8
                                            box_nms_thresh=0.7,
                                            stability_score_thresh=0.85, # 0.95
                                            crop_n_layers=1,# 0
                                            crop_n_points_downscale_factor=1,
                                            min_mask_region_area= 0,# 0
                                            )


entity_dir = args.entity_dir
view_dir = os.path.join(entity_dir, "views")

view_files = sorted(os.listdir(view_dir))

img_list = []
alpha_list = []

for data_path in tqdm(view_files):
    image_path = os.path.join(view_dir, data_path)
    image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    alpha = image_rgba[:, :, 3]

    # Ensure alpha mask is binary
    alpha[alpha < 125] = 0
    alpha[alpha >= 125] = 255

    image = cv2.imread(image_path)
    image = torch.from_numpy(image)

    img_list.append(image)
    alpha_list.append(alpha[None, ...])

# Prepare images and alphas for processing
images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
imgs = torch.cat(images)
alphas = np.concatenate(alpha_list, 0)

seg_folder = os.path.join(entity_dir, 'seg')
seg_vis_folder = os.path.join(entity_dir, 'vis_seg')
os.makedirs(seg_folder, exist_ok=True)
os.makedirs(seg_vis_folder, exist_ok=True)
# Generate segmentation maps
create(imgs, alphas, view_files, seg_folder, seg_vis_folder, mask_generator)
