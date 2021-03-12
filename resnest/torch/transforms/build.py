##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torchvision.transforms import *
from .transforms import *
from fvcore.common.registry import Registry

RESNEST_TRANSFORMS_REGISTRY = Registry('RESNEST_TRANSFORMS')

def get_transform(dataset_name):
    return RESNEST_TRANSFORMS_REGISTRY.get(dataset_name.lower())

@RESNEST_TRANSFORMS_REGISTRY.register()
def imagenet(base_size=None, crop_size=224, rand_aug=False):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    base_size = base_size if base_size is not None else int(1.0 * crop_size / 0.875)
    train_transforms = []
    val_transforms = []
    if rand_aug:
        from .autoaug import RandAugment
        train_transforms.append(RandAugment(2, 12))

    train_transforms.extend([
        ERandomCrop(crop_size),
        RandomHorizontalFlip(),
        ColorJitter(0.4, 0.4, 0.4),
        ToTensor(),
        Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
        normalize,
    ])
    val_transforms.extend([
        ECenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])
    transform_train = Compose(train_transforms)
    transform_val = Compose(val_transforms)
    return transform_train, transform_val

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
