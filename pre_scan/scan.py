'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import os
import cv2
import numpy as np
import json

def load_images(mask_path, target_size=(512, 512)):
    mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), target_size)
    return (mask > 128).astype(np.uint8)

def calculate_patch_importance(mask, patch_size, direction='horizontal'):
    H, W = mask.shape
    integral = cv2.integral(mask)
    importance = []

    outer_range = range(0, H, patch_size) if direction == 'horizontal' else range(0, W, patch_size)
    inner_range = range(0, W, patch_size) if direction == 'horizontal' else range(0, H, patch_size)

    for outer in outer_range:
        for inner in inner_range:
            i, j = (outer, inner) if direction == 'horizontal' else (inner, outer)

            x_end = min(j + patch_size, W)
            y_end = min(i + patch_size, H)

            importance_value = integral[y_end, x_end] - integral[i, x_end] - integral[y_end, j] + integral[i, j]
            importance.append(int(importance_value))  # 转换为原生 int

    return importance

def generate_EDG_scan_sequences(importance, type='start_to_end'):
    crack_indices = []
    background_indices = []
    for idx, imp in enumerate(importance):
        if imp > 0:
            crack_indices.append(idx)
        else:
            background_indices.append(idx)

    if type == 'start_to_end':
        scan_order = crack_indices + background_indices
    elif type == 'end_to_start':
        scan_order = crack_indices[::-1] + background_indices[::-1]
    else:
        raise ValueError(f"Invalid scan type: {type}")

    return scan_order, np.argsort(scan_order)

def process_single_image(mask_path, patch_size=8):
    try:
        mask_bin = load_images(mask_path)

        orders = []
        for direction in ['horizontal', 'vertical']:
            for scan_type in ['start_to_end', 'end_to_start']:
                importance = calculate_patch_importance(mask_bin, patch_size, direction)
                o, o_reversed = generate_EDG_scan_sequences(importance, scan_type)

                # 确保返回的所有值都是 Python 原生 int 类型
                orders.extend([
                    [int(i) for i in o],
                    [int(i) for i in o_reversed]
                ])
        return tuple(tuple(seq) for seq in orders)
    except Exception as e:
        print(f"Processing failed: {mask_path} - {str(e)}")
        return None

if __name__ == "__main__":
    BASE_DIR = "../../data/CrackDepth"
    DATASETS = ["train", "test", "val"]
    PATCH_SIZE = 8
    SAVE_PATH = "./scan_list/" + BASE_DIR.split('/')[-1] + f"_scan_orders_dict_patch{PATCH_SIZE}.json"

    master_dict = {dataset: {} for dataset in DATASETS}

    for dataset in DATASETS:
        dataset_dir = os.path.join(BASE_DIR, f"{dataset}_mask")

        if not os.path.exists(dataset_dir):
            print(f"Index Discontinuity or Duplication: {dataset_dir}")
            continue

        image_paths = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)

        print(f"\nDataset being processed: {dataset} ({len(image_paths)} images)")
        processed_count = 0
        for path in image_paths:
            scan_order = process_single_image(path, PATCH_SIZE)
            if scan_order:
                master_dict[dataset][path] = scan_order
                processed_count += 1

        print(f"Finish processing {dataset} dataset: Successful {processed_count}/{len(image_paths)}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'w') as f:
        json.dump(master_dict, f, indent=2)

    print(f"\nAll data has been saved to: {SAVE_PATH}")
    print("Final statistics:")
    for dataset in DATASETS:
        count = len(master_dict[dataset])
        print(f"{dataset}: Scanning order of {count} images")
