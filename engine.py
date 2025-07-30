'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

from typing import Iterable
import torch
import time
from tqdm import tqdm

def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        args=None,
        logger=None,
        scan_list=None,
):
    model.train()
    criterion.train()
    model.to(args.device)
    criterion.to(args.device)

    modal_num = len(args.modals)
    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")
    scan_orders = None

    for i, data in enumerate(data_loader):
        modal_imgs = []
        for i in range(modal_num):
            items = list(data.items())
            key, value = items[i]
            modal_imgs.append(value.to(args.device))

        targets = data['label'].to(args.device)
        if args.scan_list_json_path != 'pretrain':
            scan_list_keys_list = list(scan_list['train'].keys())
            original_scan_path = scan_list_keys_list[0].rsplit('/', 1)[0] + '/'
            path = original_scan_path + data["image_path"][-1].split('/')[-1]
            scan_orders = scan_list['train'][path]

        output_fused = model(modal_imgs, scan_orders)
        loss_final = criterion(output_fused, targets.float())
        curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        logger.info(
            f"time -> {curTime} | Epoch -> {epoch} | image_num -> {data['image_path']} | loss final -> {loss_final_str} | lr -> {l}")

        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss_final.item():.4f}")
        pbar.update(1)

    pbar.close()

