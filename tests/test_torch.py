##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import importlib
import inspect

def test_model_inference():
    # get all models
    import resnest.torch as module
    functions = inspect.getmembers(module, inspect.isfunction)
    model_list = [f[0] for f in functions]

    get_model = importlib.import_module('resnest.torch')
    x = torch.rand(1, 3, 224, 224)
    for model_name in model_list:
        print('Doing: ', model_name)
        net = getattr(get_model, model_name)
        model = net(pretrained=True)
        model.eval()
        y = model(x)

if __name__ == "__main__":
    import nose
    nose.runmodule()
