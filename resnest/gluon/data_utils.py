from PIL import Image

import mxnet as mx
from mxnet.gluon import Block
from ..transforms import *

class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = rand_augment_list()
        self.topil = ToPIL()

    def __call__(self, img):
        img = self.topil(img)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > random.uniform(0.2, 0.8):
                continue
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img


class ToPIL(object):
    """Convert image from ndarray format to PIL
    """
    def __call__(self, img):
        x = Image.fromarray(img.asnumpy())
        return x

class ToNDArray(object):
    def __call__(self, img):
        x = mx.nd.array(np.array(img), mx.cpu(0))
        return x

class AugmentationBlock(Block):
    r"""
    AutoAugment Block

    Example
    -------
    >>> from autogluon.utils.augment import AugmentationBlock, autoaug_imagenet_policies
    >>> aa_transform = AugmentationBlock(autoaug_imagenet_policies())
    """
    def __init__(self, policies):
        """
        plicies : list of (name, pr, level)
        """
        super().__init__()
        self.policies = policies
        self.topil = ToPIL()
        self.tond = ToNDArray()

    def forward(self, img):
        img = self.topil(img)
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            img = apply_augment(img, name, level)
        img = self.tond(img)
        return img
