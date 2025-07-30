import numpy as np
import torch
import torch.nn.functional as F
from mmcls.models.utils.augment.builder import AUGMENT
from .cutmix import BatchCutMixLayer
from .utils import one_hot_encoding

@AUGMENT.register_module(name='BatchResizeMix')
class BatchResizeMixLayer(BatchCutMixLayer):
    def __init__(self,
                 alpha,
                 num_classes,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 interpolation='bilinear',
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True,
                 **kwargs):
        super(BatchResizeMixLayer, self).__init__(
            alpha=alpha,
            num_classes=num_classes,
            prob=prob,
            cutmix_minmax=cutmix_minmax,
            correct_lam=correct_lam,
            **kwargs)
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.interpolation = interpolation

    def cutmix(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = lam * (self.lam_max - self.lam_min) + self.lam_min
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)

        img[:, :, bby1:bby2, bbx1:bbx2] = F.interpolate(
            img[index],
            size=(bby2 - bby1, bbx2 - bbx1),
            mode=self.interpolation)
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label
