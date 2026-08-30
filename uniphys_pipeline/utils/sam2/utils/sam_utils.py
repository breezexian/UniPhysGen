import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def resize_image(image, max_size=1280):
    # Get the current size of the image
    width, height = image.size

    # Determine the longest side
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int((max_size / width) * height)
            image = image.resize((new_width, new_height))
    else:
        if height > max_size:
            new_height = max_size
            new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height))

    return image


def seed_everything(seed_value):
    """Seeds all random number generators for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def create(image_list, alpha_list, data_list, seg_dir, seg_vis_dir, mask_generator):
    """Generates segmentation maps for each image in the list."""
    assert image_list is not None, "image_list must be provided to generate features"
    mask_generator.predictor.model.to('cuda')
    for i, img in tqdm(enumerate(image_list)):
        alpha = alpha_list[i]
        seg_path = os.path.join(seg_dir, data_list[i].split('.')[0])
        seg_vis_path = os.path.join(seg_vis_dir, data_list[i].split('.')[0])
        seg_map_vis = sam_encoder(img.unsqueeze(0), alpha, seg_path, seg_vis_path, mask_generator)

    return seg_map_vis


def save_numpy(save_path, seg_map):
    """Saves segmentation maps as numpy files."""
    save_path_s = save_path + '_s.npy'
    np.save(save_path_s, seg_map)


def get_seg_img(image, mask, bbox):
    """Extracts a segmented image using the mask and bounding box."""
    image = image.copy()
    image[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(bbox)
    return image[y:y + h, x:x + w, ...]


def pad_img(img):
    """Pads the image to make it square."""
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> list:
    """Filters masks based on the indices in `keep`."""
    keep = keep.int().cpu().numpy()
    return [m for i, m in enumerate(masks_result) if i in keep]


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """Performs non-maximum suppression on masks."""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True

    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    return idx[keep]


def masks_update(*args, **kwargs):
    """Removes redundant masks based on scores and overlap rate."""
    masks_new = ()
    for masks_lvl in (args):
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def get_location(image, foreground_mask):
    """Finds the bounding box for the largest contour in the mask."""
    contours, _ = cv2.findContours(foreground_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    return [x, y, w, h]


def vis_segmap_sam(seg_map, debug_vis_path):
    """Visualizes the segmentation map."""
    vis_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    unique_values = np.unique(seg_map)
    colors = {value: [255, 255, 255] if value == -1 else list(np.random.choice(range(256), size=3)) for value in
              unique_values}
    for key, color in colors.items():
        vis_mask[seg_map == key] = color

    cv2.imwrite(f'{debug_vis_path}_seg_map.png', vis_mask[:, :, [2, 1, 0]])
    return vis_mask[:, :, [2, 1, 0]]


def sam_encoder(image, alpha, seg_path, seg_vis_path, mask_generator):
    """Encodes the image and generates segmentation maps."""
    #vis_seg_path = save_path.replace("seg", "vis_seg")
    #os.makedirs(vis_seg_path, exist_ok=True)
    #os.makedirs(os.path.join(vis_seg_path, "part"), exist_ok=True)

    image = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    masks_m = mask_generator.generate(image)
    #masks_m = masks_update(masks_m, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    seg_map[alpha == 255] = 0

    masks_m = sorted(masks_m, key=lambda x: x['area'], reverse=True)

    for kk, mask in enumerate(masks_m):
        if kk == 0 or mask['segmentation'].sum() < 200:
            continue
        seg_map[mask['segmentation']] = kk
        # if kk in [13,12, 40, 41] and "view_4" in vis_seg_path:
        #     cur_mask = seg_map == kk
        #     cur_sum = np.sum(cur_mask)
        #     seg_sum = np.sum(mask["segmentation"])
        #     print(cur_sum, seg_sum)
        #     mask_uint8 = (mask['segmentation'].astype(np.uint8)) * 255
        #     cv2.imwrite(f"mask_{kk}_seg.png", mask_uint8)
    seg_map[alpha == 0] = -1

    for i in np.unique(seg_map):
        if i == -1:
            continue
        cur_mask = seg_map == i
        if np.sum(cur_mask) < 200:
            seg_map[cur_mask] = -1
            continue

        # bbox = get_location(image, cur_mask)
        # seg_img = get_seg_img(image, cur_mask, bbox)
        # pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
        # cv2.imwrite(f"{seg_vis_path}/part/mask_{i}.png", pad_seg_img[:, :, [2, 1, 0]])

    save_numpy(seg_path, seg_map)
    seg_map_vis = vis_segmap_sam(seg_map, seg_vis_path)
    return Image.fromarray(seg_map_vis)


def get_multi_view_mask(mult_view_dir, save_path):
    # Create plot
    fig, axes = plt.subplots(1, len(os.listdir(mult_view_dir)), figsize=(24, 8))  #
    for ind, _ in enumerate(os.listdir(mult_view_dir)):
        file_path = os.path.join(mult_view_dir, f"view_{ind}/seg_map.png")
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = axes[ind]

        # Display original image on the left
        ax.imshow(image)
        ax.set_title(f'view {ind}')
        ax.axis('off')  # Turn off axis
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # Save the image to file
    plt.savefig(f'{save_path}/gpt_mask.png')

    plt.close()


def get_multi_view_gpt_input(mult_view_dir, save_path):
    # Create plot
    fig, axes = plt.subplots(1, len(os.listdir(mult_view_dir)), figsize=(24, 8))  #
    for ind, file in enumerate(os.listdir(mult_view_dir)):
        file_path = os.path.join(mult_view_dir, f"view_{ind}.png")
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = axes[ind]

        # Display original image on the left
        ax.imshow(image)
        ax.set_title(f'view {ind}')
        ax.axis('off')  # Turn off axis

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # Save the image to file
    plt.savefig(f'{save_path}/gpt_cap.png')

    plt.close()


# root = "/home/xianzi/code/Phys-RM/render_data/bed/001/vis_seg"
# get_multi_view_mask(root, root)
# root = "/home/xianzi/code/Phys-RM/render_data/bed/001/views"
# save = "/home/xianzi/code/Phys-RM/render_data/bed/001/gpt_input"
# get_multi_view_gpt_input(root, save)
#
# root = "/home/xianzi/code/Phys-RM/render_data/bed/001/mlt_seg"
# save = "/home/xianzi/code/Phys-RM/render_data/bed/001/mlt_seg"
# get_multi_view_gpt_input(root, save)
# assert 1==2

def save_gpt_input(data_root):
    for cate in os.listdir(data_root):
        cate_dir = os.path.join(data_root, cate)
        all_cases = os.listdir(cate_dir)

        for path in all_cases:
            case_name = os.path.join(cate_dir, path)

            image_base = f"{case_name}/views"

            feature_base = f"{case_name}/seg"
            vis_seg_base = f"{case_name}/vis_seg"

            base_gpt_test_path = os.path.join(case_name, "gpt_input")
            os.makedirs(base_gpt_test_path, exist_ok=True)
            get_multi_view_gpt_input(image_base, base_gpt_test_path)
            number_view = len(os.listdir(image_base))
            for i in range(number_view):
                cur_gpt_path = os.path.join(base_gpt_test_path, f'view_{i}')
                os.makedirs(cur_gpt_path, exist_ok=True)

                img_path = os.path.join(image_base, f'view_{i}.png')
                s_path = os.path.join(feature_base, f'view_{i}_s.npy')
                seg_path = os.path.join(vis_seg_base, f'view_{i}/part')
                ss = np.load(s_path)
                rgba_image = cv2.imread(img_path)
                image = cv2.cvtColor(rgba_image, cv2.COLOR_BGR2RGB)

                mask = ss

                # Find different labels in the mask
                labels = np.unique(mask)
                labels = labels[labels != -1]  # Remove non-material parts from the mask

                # Generate random colors for each label
                colors = {}
                for label in labels:
                    # colors[label] = (random.random(), random.random(), random.random())
                    colors[label] = (1.0, 0.0, 0.0)

                # Create color mapping
                cmap = ListedColormap([colors[label] for label in labels])

                for label in labels:
                    part_image = cv2.imread(os.path.join(seg_path, f"mask_{label}.png"))
                    part_image = cv2.cvtColor(part_image, cv2.COLOR_BGR2RGB)

                    # Create plot
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # 1 row, 3 columns

                    # Display original image on the left
                    ax1.imshow(image)
                    ax1.set_title('Original Image')
                    ax1.axis('off')  # Turn off axis

                    # Display mask overlay on the middle
                    ax2.imshow(image)
                    masked_image = np.ma.masked_where(mask != label, mask)
                    ax2.imshow(masked_image, cmap=cmap, alpha=0.4, vmin=np.min(mask), vmax=np.max(mask))
                    ax2.set_title('Mask Overlay')
                    ax2.axis('off')

                    # Display part image on the right
                    ax3.imshow(part_image)
                    ax3.set_title('Part Image')
                    ax3.axis('off')

                    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

                    # Save the image to file
                    plt.savefig(f'{cur_gpt_path}/{label}.png')

                    plt.close()
