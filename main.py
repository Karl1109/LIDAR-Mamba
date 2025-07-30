'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
import json

def get_args_parser():
    parser = argparse.ArgumentParser('LIDAR FOR MULTI-MODAL CRACK SEGMENTATION', add_help=False)
    parser.add_argument('--BCELoss_ratio', default=1, type=float)
    parser.add_argument('--DiceLoss_ratio', default=1, type=float)
    parser.add_argument('--dataset_path', default="../data/CrackDepth", help='path to images')
    parser.add_argument('--scan_list_json_path', default="./pre_scan/scan_list/CrackDepth_scan_orders_dict_patch8.json",
                        help='path to images. If you want to pretrain to generate the mask, change its value to \'pretrain\'')
    # parser.add_argument('--scan_list_json_path', default="pretrain", help='path to scan lists')
    parser.add_argument('--inference_mask', default=False,
                        help='Inference mask. If you want to pretrain to generate the mask, change its value to \'True\'')
    parser.add_argument('--batch_size_train', type=int, default=1, help='train input batch size')
    parser.add_argument('--batch_size_test', type=int, default=1, help='test input batch size')
    parser.add_argument('--modals', nargs='+', default=['RGB', 'dep'], help='modal name for process')
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='./checkpoints/weights', help='save path for weights')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_mode', type=str, default='crack')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches,  takes them randomly')
    parser.add_argument('--num_threads', default=1, type=int, help='num_workers')
    parser.add_argument('--phase', type=str, default='train', help='train, test, etc')
    parser.add_argument('--load_width', type=int, default=512, help='load image width')
    parser.add_argument('--load_height', type=int, default=512, help='load image height')

    return parser

def main(args):
    checkpoints_path = "./checkpoints"
    curTime = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    modal_num = len(args.modals)
    modals_name = ''
    for i in range(len(args.modals)):
        modals_name = modals_name + '_' + (args.modals)[i]

    if args.scan_list_json_path == 'pretrain':
        process_floder_path = os.path.join(checkpoints_path, curTime + '_Dataset->' + dataset_name + '_modals->' + modals_name + '_pretrain')
    else:
        process_floder_path = os.path.join(checkpoints_path, curTime + '_Dataset->' + dataset_name + '_modals->' + modals_name)
    if not os.path.exists(process_floder_path):
        os.makedirs(process_floder_path)
    else:
        print("create process folder error!")

    log_train = get_logger(process_floder_path, 'train')
    log_test = get_logger(process_floder_path, 'test')
    log_eval = get_logger(process_floder_path, 'eval')

    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("processing modal -> " + str(args.modals))
    log_train.info("number of modal -> " + str(modal_num))
    print('processing modal -> ', args.modals)
    print('number of modal -> ', modal_num)

    if args.scan_list_json_path == 'pretrain':
        args.epochs = 10
        print("pretrain! epoches -> " + str(args.epochs))
        log_train.info("pretrain! epoches -> " + str(args.epochs))

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()],
            "lr": args.lr,
        },
    ]

    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)

    output_dir = args.output_dir + '/' + curTime + '_Dataset->' + dataset_name + '_modals->' + modals_name
    if args.output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time()

    scan_list = None
    if args.scan_list_json_path != 'pretrain':
        scan_list_json_path = args.scan_list_json_path
        with open(scan_list_json_path, 'r') as f:
            scan_list = json.load(f)
            log_train.info("Load scan list success! ")
            print("Load scan list success! ")

    max_F1 = 0
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        args.phase = 'train'
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)

        train_one_epoch(
            model, criterion, train_dataLoader, optimizer, epoch, args, log_train, scan_list)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("testing epoch start -> ", epoch)
        results_path = curTime + '_Dataset->' + dataset_name + '_modal->' + modals_name
        save_root = f'./results/{results_path}/results_' + str(epoch)
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        scan_orders = None
        if args.scan_list_json_path != 'pretrain':
            scan_list_keys_list = list(scan_list['test'].keys())
            original_scan_path = scan_list_keys_list[0].rsplit('/', 1)[0] + '/'

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            for batch_idx, (data) in enumerate(test_dl):
                modal_imgs = []
                for i in range(modal_num):
                    items = list(data.items())
                    key, value = items[i]
                    modal_imgs.append(value.to(torch.device(args.device)))

                target = data["label"]
                target = target.to(dtype=torch.int64).cuda()

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

                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)

        pbar.close()
        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("testing epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("evaluating epoch start -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        if(max_F1 < metrics['F1']):
            max_Metrics = metrics
            max_F1 = metrics['F1']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evaluating epoch finish -> ", epoch)
        print('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evaluating epoch finish -> " + str(epoch))
        log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_F1 -> ' + str(max_Metrics['F1']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LIDAR FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
