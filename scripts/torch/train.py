##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from resnest.torch.config import get_cfg
from resnest.torch.models.build import get_model
from resnest.torch.datasets import get_dataset
from resnest.torch.transforms import get_transform
from resnest.torch.loss import get_criterion
from resnest.torch.utils import (save_checkpoint, accuracy,
        AverageMeter, LR_Scheduler, torch_dist_sum)

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='ResNeSt Training')
        parser.add_argument('--config-file', type=str, default=None,
                            help='training configs')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        # evaluation option
        parser.add_argument('--eval-only', action='store_true', default= False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def main():
    args = Options().parse()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    # load config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.OPTIMIZER.LR = cfg.OPTIMIZER.LR * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print(f'rank: {args.rank} / {args.world_size}')
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    if args.gpu == 0:
        print(args)

    # init the global
    global best_pred, acclist_train, acclist_val

    # seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # init dataloader
    transform_train, transform_val = get_transform(cfg.DATASET.NAME)(
            cfg.DATA.BASE_SIZE, cfg.DATA.CROP_SIZE, cfg.DATA.RAND_AUG)
    trainset = get_dataset(cfg.DATASET.NAME)(root=cfg.DATA.ROOT,
                                             transform=transform_train,
                                             train=True,
                                             download=True)
    valset = get_dataset(cfg.DATASET.NAME)(root=cfg.DATA.ROOT,
                                           transform=transform_val,
                                           train=False,
                                           download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAINING.WORKERS, pin_memory=True,
        sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.TRAINING.TEST_BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAINING.WORKERS, pin_memory=True,
        sampler=val_sampler)
    
    # init the model
    model_kwargs = {}
    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    model = get_model(args.model)(**model_kwargs)

    if args.gpu == 0:
        print(model)

    criterion, train_loader = get_criterion(cfg, train_loader, args.gpu)

    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])

    # criterion and optimizer
    if cfg.OPTIMIZER.DISABLE_BN_WD:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        if args.gpu == 0:
            print(" Weight decay NOT applied to BN parameters ")
            print(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0 },
                                     {'params': rest_params, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}],
                                    lr=cfg.OPTIMIZER.LR,
                                    momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.OPTIMIZER.LR,
                                    momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0:
                print("=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            cfg.TRAINING.START_EPOCHS = checkpoint['epoch'] + 1 if cfg.TRAINING.START_EPOCHS == 0 else args.start_epoch
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.gpu == 0:
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise RuntimeError (f"=> no resume checkpoint found at '{args.resume}'")
    scheduler = LR_Scheduler(cfg.OPTIMIZER.LR_SCHEDULER,
                             base_lr=cfg.OPTIMIZER.LR,
                             num_epochs=cfg.TRAINING.EPOCHS,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPTIMIZER.WARMUP_EPOCHS)
    def train(epoch):
        train_sampler.set_epoch(epoch)
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        for batch_idx, (data, target) in enumerate(train_loader):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            if not cfg.DATA.MIXUP:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if not cfg.DATA.MIXUP:
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

            losses.update(loss.item(), data.size(0))
            if batch_idx % 100 == 0 and args.gpu == 0:
                if cfg.DATA.MIXUP:
                    print('Batch: %d| Loss: %.3f'%(batch_idx, losses.avg))
                else:
                    print('Batch: %d| Loss: %.3f | Top1: %.3f'%(batch_idx, losses.avg, top1.avg))

        acclist_train += [top1.avg]

    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            with torch.no_grad():
                output = model(data)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        # sum all
        sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.gpu, top1.sum, top1.count, top5.sum, top5.count)

        if args.gpu == 0:
            top1_acc = sum(sum1) / sum(cnt1)
            top5_acc = sum(sum5) / sum(cnt5)
            print('Validation: Top1: %.3f | Top5: %.3f'%(top1_acc, top5_acc))
            if args.eval_only:
                return

            # save checkpoint
            acclist_val += [top1_acc]
            if top1_acc > best_pred:
                best_pred = top1_acc 
                is_best = True
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_pred,
                    'acclist_train':acclist_train,
                    'acclist_val':acclist_val,
                },
                args=args,
                cfg=cfg,
                is_best=is_best)

    if args.export:
        if args.gpu == 0:
            torch.save(model.module.state_dict(), args.export + '.pth')
        return

    if args.eval_only:
        validate(cfg.TRAINING.START_EPOCHS)
        return

    for epoch in range(cfg.TRAINING.START_EPOCHS, cfg.TRAINING.EPOCHS):
        tic = time.time()
        train(epoch)
        if epoch % 10 == 0 or epoch == cfg.TRAINING.EPOCHS - 1:
            validate(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            print(f'Epoch: {epoch}, Time cost: {elapsed}')

    if args.gpu == 0:
        save_checkpoint({
                'epoch': cfg.TRAINING.EPOCHS - 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'acclist_train':acclist_train,
                'acclist_val':acclist_val,
            },
            args=args,
            cfg=cfg,
            is_best=False)

if __name__ == "__main__":
    main()
