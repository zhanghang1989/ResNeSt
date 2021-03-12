
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import argparse, os, math, time, sys

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

from resnest.gluon import get_model

from PIL import Image

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.encoding/data/ILSVRC2012/',
                        help='Imagenet directory for validation.')
    parser.add_argument('--rec-dir', type=str, default=None,
                        help='recio directory for validation.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=32, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--model', type=str, default='model', required=False,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='input shape of the image, default is 224.')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='The ratio for crop and input size, for validation dataset only')
    parser.add_argument('--dtype', type=str,
                        help='training data type')
    parser.add_argument('--dilation', type=int, default=1,
                        help='network dilation. default 1 (no-dilation)')
    opt = parser.parse_args()
    return opt

def test(network, ctx, val_data, batch_fn):
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    num_batch = len(val_data)
    num = 0
    start = time.time()
 
    iterator = enumerate(val_data)
    next_i, next_batch = next(iterator)
    next_data, next_label = batch_fn(next_batch, ctx)
    stop = False
    while not stop:
        i = next_i
        data = next_data
        label = next_label
        outputs = [network(X.astype(opt.dtype, copy=False)) for X in data]
        try:
            next_i, next_batch = next(iterator)
            next_data, next_label = batch_fn(next_batch, ctx)
            if next_i == 5:
                # warm-up
                num = 0
                mx.nd.waitall()
                start = time.time()
        except StopIteration:
            stop = True
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        print('%d / %d : %.8f, %.8f'%(i, num_batch, 1-top1, 1-top5))
        num += batch_size

    end = time.time()
    speed = num / (end - start)
    print('Throughput is %f img/sec.'% speed)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)


if __name__ == '__main__':
    opt = parse_args()

    batch_size = opt.batch_size
    classes = 1000

    num_gpus = opt.num_gpus
    if num_gpus > 0:
        batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    input_size = opt.crop_size
    model_name = opt.model
    pretrained = True if not opt.resume else False

    kwargs = {'ctx': ctx, 'pretrained': pretrained, 'classes': classes}

    if opt.dilation > 1:
        kwargs['dilation'] = opt.dilation

    net = get_model(model_name, **kwargs)
    net.cast(opt.dtype)
    if opt.resume:
        net.load_parameters(opt.resume, ctx=ctx)
    else:
        net.hybridize()

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size/crop_ratio))

    if input_size >= 320:
        from resnest.gluon.transforms import ECenterCrop
        from resnest.gluon.data_utils import ToPIL, ToNDArray
        transform_test = transforms.Compose([
            ToPIL(),
            ECenterCrop(input_size),
            ToNDArray(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

    if not opt.rec_dir:
        from gluoncv.data import imagenet
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        imgrec = os.path.join(opt.rec_dir, 'val.rec')
        imgidx = os.path.join(opt.rec_dir, 'val.idx')
        val_data = gluon.data.DataLoader(
            mx.gluon.data.vision.ImageRecordDataset(imgrec).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    err_top1_val, err_top5_val = test(net, ctx, val_data, batch_fn)
    print(err_top1_val, err_top5_val)
