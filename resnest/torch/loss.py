import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from resnest.torch.utils import MixUpWrapper

__all__ = ['LabelSmoothing', 'NLLMultiLabelSmooth', 'get_criterion']

def get_criterion(cfg, train_loader, gpu):
    if cfg.DATA.MIXUP > 0:
        train_loader = MixUpWrapper(cfg.DATA.MIXUP, 1000, train_loader, gpu)
        criterion = NLLMultiLabelSmooth(cfg.DATA.LABEL_SMOOTHING)
    elif cfg.DATA.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothing(cfg.DATA.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion, train_loader

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
    
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)
    
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

