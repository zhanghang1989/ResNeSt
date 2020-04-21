##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Ablation Study Models for ResNeSt"""
from .resnet import ResNet, Bottleneck
from mxnet import cpu

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']

def resnest50_fast_1s1x64d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, cardinality=1, bottleneck_width=64,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_1s1x64d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_2s1x64d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, cardinality=1, bottleneck_width=64,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_2s1x64d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_4s1x64d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, cardinality=1, bottleneck_width=64,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_4s1x64d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_1s2x40d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, cardinality=2, bottleneck_width=40,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_1s2x40d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_2s2x40d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, cardinality=2, bottleneck_width=40,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_2s2x40d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_4s2x40d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, cardinality=2, bottleneck_width=40,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_4s2x40d',
                                             root=root), ctx=ctx)
    return model

def resnest50_fast_1s4x24d(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, cardinality=4, bottleneck_width=24,
                   deep_stem=True, avg_down=True,
                   avd=True, avd_first=True,
                   use_splat=True, dropblock_prob=0.1,
                   name_prefix='resnetv1f_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50_fast_1s4x24d',
                                             root=root), ctx=ctx)
    return model

