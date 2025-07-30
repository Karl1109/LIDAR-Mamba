'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

from thop import profile
import torch
from main import get_args_parser
import argparse
from models.Decoder import build

parser = argparse.ArgumentParser('LIDAR FOR CRACK', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    args.scan_list_json_path = 'pretrain'
    args.modals = ['RGB', 'dep']
    model, _, = build(args)
    model.to(args.device)

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total Parameters: {total_params}")

    input = []
    input1 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))

    for i in range(len(args.modals)):
        input.append(input1)

    flops, params = profile(model, (input, None))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
