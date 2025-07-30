import random
import numpy as np
from .builder import build_augment

class Augments(object):
    def __init__(self, augments_cfg):
        super(Augments, self).__init__()

        if isinstance(augments_cfg, dict):
            augments_cfg = [augments_cfg]

        assert len(augments_cfg) > 0, \
            'The length of augments_cfg should be positive.'
        self.augments = [build_augment(cfg) for cfg in augments_cfg]
        self.augment_probs = [aug.prob for aug in self.augments]

        has_identity = any([cfg['type'] == 'Identity' for cfg in augments_cfg])
        if has_identity:
            assert sum(self.augment_probs) == 1.0,\
                'The sum of augmentation probabilities should equal to 1,' \
                ' but got {:.2f}'.format(sum(self.augment_probs))
        else:
            assert sum(self.augment_probs) <= 1.0,\
                'The sum of augmentation probabilities should less than or ' \
                'equal to 1, but got {:.2f}'.format(sum(self.augment_probs))
            identity_prob = 1 - sum(self.augment_probs)
            if identity_prob > 0:
                num_classes = self.augments[0].num_classes
                self.augments += [
                    build_augment(
                        dict(
                            type='Identity',
                            num_classes=num_classes,
                            prob=identity_prob))
                ]
                self.augment_probs += [identity_prob]

    def __call__(self, img, gt_label):
        if self.augments:
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            aug = random_state.choice(self.augments, p=self.augment_probs)
            return aug(img, gt_label)
        return img, gt_label
