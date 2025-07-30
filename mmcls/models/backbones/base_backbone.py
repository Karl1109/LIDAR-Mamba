from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule
class BaseBackbone(BaseModule, metaclass=ABCMeta):

    def __init__(self, init_cfg=None):
        super(BaseBackbone, self).__init__(init_cfg)

    @abstractmethod
    def forward(self, x):
        pass

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
