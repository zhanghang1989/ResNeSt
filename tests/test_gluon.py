##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import mxnet as mx

def test_model_inference():
    # get all models
    from resnest.gluon.model_store import _model_sha1
    from resnest.gluon import get_model

    model_list = _model_sha1.keys()

    x = mx.random.uniform(shape=(1, 3, 224, 224))
    for model_name in model_list:
        print('Doing: ', model_name)
        model = get_model(model_name, pretrained=True)
        y = model(x)

if __name__ == "__main__":
    import nose
    nose.runmodule()

