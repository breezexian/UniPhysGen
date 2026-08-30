import os
import random
import argparse
import numpy as np
from io import BytesIO
import matplotlib as mpl
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# Add custom font
font_path = 'times-new-roman.ttf'  # Replace with the path to the font file
fm.fontManager.addfont(font_path)



def parse_txt_file(file_path):
    parsed_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 5:
                    file_path, description, material, hardness, AD_type = parts
                    # Extract 00 and 01 from the file path
                    path_parts = file_path.split('/')
                    image_number = path_parts[-2]
                    part_number = path_parts[-1]
                    parsed_data.append({
                        "image_number": image_number,
                        "part_number": part_number[:2],
                        "material": material.strip().lower().replace(" ", "")
                    })

    return parsed_data

def cat_rgba(image1, image2):
    # Ensure both images have the same height for horizontal concatenation
    max_height = max(image1.height, image2.height)

    # Create a new image with width as the sum of both images' widths, and height as the maximum height
    new_image = Image.new('RGBA', (image1.width + image2.width, max_height))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))  # Paste image1 at (0, 0)
    new_image.paste(image2, (image1.width, 0))  # Paste image2 to the right of image1

    
    return new_image


def filter_and_process(parsed_data, target_image_number):
    filtered_data = [item for item in parsed_data if int(item['image_number']) == target_image_number]

    # Find the highest part number
    max_part_number = max(int(item['part_number']) for item in filtered_data)

    # Create a list of materials with None for missing parts
    materials_list = [None] * (max_part_number + 1)
    for item in filtered_data:
        part_index = int(item['part_number'])
        materials_list[part_index] = item['material']

    return materials_list

def visualize_and_save_segmentation(original_image_path, mask_image_path, output_path, m_label, alpha=0.5):
    # Load the original image and mask
    original_image = np.array(Image.open(original_image_path).convert('RGBA'))
    mask_image = np.load(mask_image_path)  # Load mask as grayscale
    seg_map_vis = vis_segmap(mask_image)
    # Get the tab10 colormap
    cmap_tab10 = mpl.colormaps['tab10']

    # Create a color segmentation map with an alpha channel
    segmentation_map = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)

    unique_labels = np.unique(mask_image)
    for label in unique_labels:
        if label == -1:
            segmentation_map[mask_image == label] = [0, 0, 0, 0]  # Transparent background
        else:
            color = np.array(cmap_tab10(m_label[label] % 10)[:3]) * 255
            segmentation_map[mask_image == label, :3] = color
            segmentation_map[mask_image == label, 3] = 255  # Fully opaque for labeled regions

    # Blend the original image with the segmentation map using the alpha parameter
    blended_image = np.zeros_like(original_image, dtype=np.uint8)
    for i in range(4):  # Loop through RGBA channels
        blended_image[:, :, i] = (original_image[:, :, i] * (1 - alpha) + 
                                  segmentation_map[:, :, i] * alpha).astype(np.uint8)

    # Create an empty image with an alpha channel for the combined image
    combined_image_with_alpha = np.zeros((original_image.shape[0], original_image.shape[1] * 3, 4), dtype=np.uint8)

    # Add the original image to the left side of the combined image
    combined_image_with_alpha[:, :original_image.shape[1], :] = original_image

    # Add the seg image to the right side of the combined image
    combined_image_with_alpha[:, original_image.shape[1]:2*original_image.shape[1], :] = seg_map_vis

    # Add the blended image to the right side of the combined image
    combined_image_with_alpha[:, 2*original_image.shape[1]:, :] = blended_image

    # Save the combined image with transparency
    plt.imsave(output_path, combined_image_with_alpha)
    return Image.fromarray(combined_image_with_alpha)


def parse_material(materials):
    material_mapping = {}
    counter = 0

    # Convert the list
    converted_list = []
    m_name_index = []
    for material in materials:
        if material is None or material == "-1":
            converted_list.append(-1)
        else:
            if material not in material_mapping:
                print(material)
                material_mapping[material] = counter
                m_name_index.append(material)
                counter += 1
            converted_list.append(material_mapping[material])
    return converted_list, m_name_index



def make_legend(colors, names, ncol=1, figsize=(2.0, 2.5), savefile=None, show=False):
    plt.style.use('fast')
    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    plt.axis('off')
    
    # Create legend with color boxes
    ptchs = []
    for color, name in zip(colors, names):
        print(name)
        if len(name) > 10:  # Wrap long names
            name = name.replace(' ', '\n')
        ptchs.append(mpatches.Patch(color=color[:3], label=name))
    
    # Add legend to the figure
    leg = plt.legend(handles=ptchs, ncol=ncol, loc='center left', prop={'size': 18}, 
                     handlelength=1, handleheight=1, facecolor='white', framealpha=0)
    plt.tight_layout()

    # Save the figure to a BytesIO object
    img_bytes = BytesIO()
    if savefile is not None:
        plt.savefig(savefile, dpi=400)

    plt.savefig(img_bytes, format='png', dpi=400)
    img_bytes.seek(0)  # Rewind the buffer

    # Convert the image to a PIL Image object
    img = Image.open(img_bytes)

    if show:
        plt.show()
    plt.close()

    return img  # Return the image as a PIL Image object

    


def vis_segmap(seg_map):
    # Create a blank image
    vis_mask = np.zeros((seg_map.shape[0], seg_map.shape[1], 4), dtype=np.uint8)

    # Get unique values in the mask
    unique_values = np.unique(seg_map)

    # Create color mapping
    colors = {}
    for value in unique_values:
        if value == -1:
            colors[value] = [0, 0, 0, 0]  # Transparent background
        elif value >= 0:
            random_color = np.random.choice(range(256), size=3)
            colors[value] = list([random_color[0], random_color[1], random_color[2], 255])  # Assign random colors

    # Map different numbers to different colors
    for key in colors.keys():
        vis_mask[seg_map == key] = colors[key]
    img = Image.fromarray(vis_mask)
    # img.save("debug.png", "PNG")
    return img
