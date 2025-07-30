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
import json

parser = argparse.ArgumentParser('LIDAR FOR MULTI-MODAL CRACK SEGMENTATION', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
args.dataset_path = '../data/CrackDepth'

if __name__ == '__main__':
    args.batch_size = 1
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    load_model_file = "./checkpoints/weights/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    modal_num = len(args.modals)

    if args.scan_list_json_path != 'pretrain':
        scan_list_json_path = args.scan_list_json_path
        with open(scan_list_json_path, 'r') as f:
            scan_list = json.load(f)
            print("Load scan list success! ")
            scan_list_keys_list = list(scan_list['test'].keys())
            original_scan_path = scan_list_keys_list[0].rsplit('/', 1)[0] + '/'

    model.cuda()
    save_root = "./results/results_TEST"
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_dl):

            modal_imgs = []
            for i in range(modal_num):
                items = list(data.items())
                key, value = items[i]
                modal_imgs.append(value.to(args.device))

            target = data["label"].cuda()
            if args.scan_list_json_path != 'pretrain':
                path = original_scan_path + data["image_path"][-1].split('/')[-1]
                scan_orders = scan_list['test'][path]

            out = model(modal_imgs, scan_orders)

            loss = criterion(out, target.float())
            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["image_path"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            print('----------------------------------------------------------------------------------------------')
            print("loss -> ", loss)
            print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("All Test Finished!")

