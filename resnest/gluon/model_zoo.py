##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from .resnest import *
from .ablation import *

_all__ = ['get_model', 'get_model_list']

models = {
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269,
    'resnest50_fast_1s1x64d': resnest50_fast_1s1x64d,
    'resnest50_fast_2s1x64d': resnest50_fast_2s1x64d,
    'resnest50_fast_4s1x64d': resnest50_fast_4s1x64d,
    'resnest50_fast_1s2x40d': resnest50_fast_1s2x40d,
    'resnest50_fast_2s2x40d': resnest50_fast_2s2x40d,
    'resnest50_fast_4s2x40d': resnest50_fast_4s2x40d,
    'resnest50_fast_1s4x24d': resnest50_fast_1s4x24d,
    }

def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    Module:
        The model.
    """

    name = name.lower()
    if name in models:
        net = models[name](**kwargs)
    else:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    return net

def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return models.keys()

