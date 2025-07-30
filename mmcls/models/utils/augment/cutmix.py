from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from .builder import AUGMENT
from .utils import one_hot_encoding

class BaseCutMixLayer(object, metaclass=ABCMeta):

    def __init__(self,
                 alpha,
                 num_classes,
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True):
        super(BaseCutMixLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob
        self.cutmix_minmax = cutmix_minmax
        self.correct_lam = correct_lam

    def rand_bbox_minmax(self, img_shape, count=None):
        assert len(self.cutmix_minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(
            int(img_h * self.cutmix_minmax[0]),
            int(img_h * self.cutmix_minmax[1]),
            size=count)
        cut_w = np.random.randint(
            int(img_w * self.cutmix_minmax[0]),
            int(img_w * self.cutmix_minmax[1]),
            size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    def rand_bbox(self, img_shape, lam, margin=0., count=None):
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def cutmix_bbox_and_lam(self, img_shape, lam, count=None):
        if self.cutmix_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape, count=count)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam, count=count)
        if self.correct_lam or self.cutmix_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    @abstractmethod
    def cutmix(self, imgs, gt_label):
        pass


@AUGMENT.register_module(name='BatchCutMix')
class BatchCutMixLayer(BaseCutMixLayer):

    def __init__(self, *args, **kwargs):
        super(BatchCutMixLayer, self).__init__(*args, **kwargs)

    def cutmix(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)
        img[:, :, bby1:bby2, bbx1:bbx2] = \
            img[index, :, bby1:bby2, bbx1:bbx2]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.cutmix(img, gt_label)
