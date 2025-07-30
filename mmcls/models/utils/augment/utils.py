import torch.nn.functional as F

def one_hot_encoding(gt, num_classes):
    if gt.ndim == 1:
        return F.one_hot(gt, num_classes=num_classes)
    else:
        return gt
