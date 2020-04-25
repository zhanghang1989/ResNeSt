##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt ablation study models"""

import torch
from .resnet import ResNet, Bottleneck

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('d8fbf808', 'resnest50_fast_1s1x64d'),
    ('44938639', 'resnest50_fast_2s1x64d'),
    ('f74f3fc3', 'resnest50_fast_4s1x64d'),
    ('32830b84', 'resnest50_fast_1s2x40d'),
    ('9d126481', 'resnest50_fast_2s2x40d'),
    ('41d14ed0', 'resnest50_fast_4s2x40d'),
    ('d4a4f76f', 'resnest50_fast_1s4x24d'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50_fast_1s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_1s1x64d'], progress=True, check_hash=True))
    return model

def resnest50_fast_2s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_2s1x64d'], progress=True, check_hash=True))
    return model

def resnest50_fast_4s1x64d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_4s1x64d'], progress=True, check_hash=True))
    return model

def resnest50_fast_1s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_1s2x40d'], progress=True, check_hash=True))
    return model

def resnest50_fast_2s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_2s2x40d'], progress=True, check_hash=True))
    return model

def resnest50_fast_4s2x40d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_4s2x40d'], progress=True, check_hash=True))
    return model

def resnest50_fast_1s4x24d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=4, bottleneck_width=24,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50_fast_1s4x24d'], progress=True, check_hash=True))
    return model
