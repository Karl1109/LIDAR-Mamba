'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import numpy as np
import torch
import argparse
import os
import cv2
from datasets import create_dataset
from models import build_model
from main import get_args_parser

parser = argparse.ArgumentParser('LIDAR FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'

if __name__ == '__main__':
    args.batch_size = 1
    args.scan_list_json_path = 'pretrain'
    args.modals = ['RGB', 'dep']
    device = torch.device(args.device)

    load_model_file = "./checkpoints/weights/checkpoint_best.pth"
    model, criterion = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    modal_num = len(args.modals)

    # Change the value of 'dataset_root' to point to the path of the dataset
    dataset_root = '../data/CrackDepth'

    phases = ['train', 'val', 'test']
    for phase in phases:
        print(f'\n{"=" * 40} Processing {phase} set {"=" * 40}')

        args.phase = phase
        args.dataset_path = os.path.join(dataset_root)
        save_root = os.path.join(dataset_root, f'{phase}_mask')
        args.inference_mask = True
        test_dl = create_dataset(args)
        os.makedirs(save_root, exist_ok=True)
        data_size = len(test_dl)

        with torch.no_grad():
            for batch_idx, (data) in enumerate(test_dl):
                modal_imgs = []
                for i in range(modal_num):
                    items = list(data.items())
                    key, value = items[i]
                    modal_imgs.append(value.to(args.device))

                target = data["label"].cuda()
                out = model(modal_imgs, scan_orders=None)
                loss = criterion(out, target.float())

                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["image_path"][0].split("/")[-1][0:-4]

                out = np.where(out >= 0.5, 255, 0).astype(np.uint8)
                cv2.imwrite(os.path.join(save_root, f"{root_name}.png"), out)
                print(f'Processed {batch_idx + 1}/{data_size} | Loss: {loss.item():.4f}')

    print("\nAll Inference Completed!")