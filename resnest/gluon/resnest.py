##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt implemented in Gluon."""

__all__ = ['resnest50', 'resnest101',
           'resnest200', 'resnest269']

from .resnet import ResNet, Bottleneck
from mxnet import cpu

def resnest50(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                      radix=2, cardinality=1, bottleneck_width=64,
                      deep_stem=True, avg_down=True,
                      avd=True, avd_first=False,
                      use_splat=True, dropblock_prob=0.1,
                      name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50', root=root), ctx=ctx)
    return model

def resnest101(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                      radix=2, cardinality=1, bottleneck_width=64,
                      deep_stem=True, avg_down=True, stem_width=64,
                      avd=True, avd_first=False, use_splat=True, dropblock_prob=0.1,
                      name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest101', root=root), ctx=ctx)
    return model

def resnest200(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], deep_stem=True, avg_down=True, stem_width=64,
                      avd=True, use_splat=True, dropblock_prob=0.1, final_drop=0.2,
                      name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest200', root=root), ctx=ctx)
    return model

def resnest269(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8], deep_stem=True, avg_down=True, stem_width=64,
                      avd=True, use_splat=True, dropblock_prob=0.1, final_drop=0.2,
                      name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest269', root=root), ctx=ctx)
    return model
