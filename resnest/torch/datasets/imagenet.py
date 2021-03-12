##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

from .build import RESNEST_DATASETS_REGISTRY

@RESNEST_DATASETS_REGISTRY.register()
class ImageNet(datasets.ImageFolder):
    def __init__(self, root=os.path.expanduser('~/.encoding/data/ILSVRC2012'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split='train' if train == True else 'val'
        root = os.path.join(root, split)
        super().__init__(root, transform, target_transform)
