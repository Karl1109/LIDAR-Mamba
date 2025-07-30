'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import os
import json
import random
from scan import process_single_image

def validate_json_structure(master_data):
    assert isinstance(master_data, dict), "The JSON root element should be a dictionary"
    required_datasets = ["train", "test", "val"]

    for dataset in required_datasets:
        assert dataset in master_data, f"Missing required datasets: {dataset}"
        assert isinstance(master_data[dataset], dict), f"The value of dataset {dataset} is not a dictionary."

    total_images = 0
    for dataset, dataset_data in master_data.items():
        print(f"\nValidating dataset: {dataset}")
        assert len(dataset_data) > 0, f"{dataset} is empty"

        img_count = 0
        for img_path, orders in dataset_data.items():
            assert os.path.exists(img_path), f"Not exist: {img_path}"

            assert len(orders) == 8, f"There should be 8 scanning orders, which actually gives {len(orders)}"
            for seq in orders:
                assert len(seq) == 4096, f"Wrong number of indexes: {len(seq)}"
                assert set(seq) == set(range(4096)), "Index Discontinuity or Duplication"

            img_count += 1
            total_images += 1

        print(f"Dataset {dataset} validated with {img_count} images")

    print(f"\nStructure validation completed, total verified images: {total_images}")


def sample_recheck(master_data, sample_size=3, patch_size=8):
    candidate_samples = []
    for dataset, dataset_data in master_data.items():
        candidate_samples.extend([(dataset, path) for path in dataset_data.keys()])

    selected_samples = random.sample(candidate_samples, min(sample_size, len(candidate_samples)))
    for dataset, img_path in selected_samples:
        print(f"\nRechecking {dataset} dataset: {img_path}")
        original_orders = [tuple(seq) for seq in master_data[dataset][img_path]]
        regenerated = process_single_image(img_path, patch_size)
        assert regenerated is not None, "FAILED!"
        regenerated_orders = [tuple(seq) for seq in regenerated]

        mismatch_found = False
        for order_idx, (orig, regen) in enumerate(zip(original_orders, regenerated_orders)):
            if orig != regen:
                print(f"Order {order_idx} mismatch! Sample comparison:")
                print(f"Original first 5 indices: {orig[:5]}")
                print(f"Regenerated first 5 indices: {regen[:5]}")
                mismatch_found = True
                break

        if not mismatch_found:
            print(f"All scan orders verified consistent (total {len(original_orders)} orders)")


def run_all_tests(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file is not exist: {json_path}")

    with open(json_path, 'r') as f:
        master_data = json.load(f)

    print("=== Starting structure validation ===")
    validate_json_structure(master_data)
    print("\n=== Starting cross-dataset sampling recheck ===")
    sample_recheck(master_data, sample_size=5)
    print("\n=== All tests passed ===")


if __name__ == "__main__":
    # Change the value of 'JSON_PATH' to point to the generated JSON file
    JSON_PATH = "scan_list/CrackDepth_scan_orders_dict_patch8.json"

    try:
        run_all_tests(JSON_PATH)
    except AssertionError as ae:
        print(f"\n!!! Assertion failed: {str(ae)}")
    except Exception as e:
        print(f"\n!!! Unhandled exception: {str(e)}")