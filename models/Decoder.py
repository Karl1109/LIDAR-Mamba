'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui1109@stud.tjut.edu.cn, liuhui@ieee.org
'''

import torch
from torch import nn
from mmcls.LIDAR_dev.models.LIDAR.LIDAR import LIDAR
from models.LD3CF import LD3CF

class Decoder(nn.Module):
    def __init__(self, backbones, args=None):
        super().__init__()
        self.args = args
        self.backbones = nn.ModuleList(backbones)
        num_modalities = len(backbones)
        self.LD3CF = LD3CF(num_modalities, embedding_dim = 8)

    def forward(self, modal_imgs, scan_orders):
        backbone_outs = []
        for i in range(len(self.backbones)):
            backbone_out = self.backbones[i](modal_imgs[i], scan_orders)
            backbone_outs.append(backbone_out)

        out = self.LD3CF(backbone_outs)
        return out

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class BCE_Dice(nn.Module):
    def __init__(self, args):
        super(BCE_Dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice

def build(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)
    modals = args.modals

    backbones = []
    for i in range(len(modals)):
        backbone = LIDAR(arch='Crack', out_indices=(0, 1, 2, 3),
                           drop_path_rate=0.2,
                           final_norm=True,
                           convert_syncbn=True)

        backbones.append(backbone.cuda())

    model = Decoder(backbones, args)
    criterion = BCE_Dice(args)
    criterion.to(device)

    return model, criterion


