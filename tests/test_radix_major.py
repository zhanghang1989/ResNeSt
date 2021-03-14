import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

from resnest.torch.models.splat import SplAtConv2d, DropBlock2D

class RadixMajorNaiveImp(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(RadixMajorNaiveImp, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        assert not self.use_bn

        self.relu = ReLU(inplace=True)
        cardinal_group_width = channels // groups
        cardinal_inter_channels = inter_channels // groups

        self.fc1 = nn.ModuleList([nn.Linear(cardinal_group_width, cardinal_inter_channels) for _ in range(groups)])
        self.fc2 = nn.ModuleList([nn.Linear(cardinal_inter_channels, cardinal_group_width*radix) for _ in range(groups)])

        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        cardinality = self.cardinality
        radix = self.radix

        tiny_group_width = channel//radix//cardinality
        all_groups = torch.split(x, tiny_group_width, dim=1)

        out = []
        for k in range(cardinality):
            U_k = [all_groups[r * cardinality + k] for r in range(radix)]
            U_k = sum(U_k)
            gap_k = F.adaptive_avg_pool2d(U_k, 1).squeeze()
            atten_k = self.fc2[k](self.fc1[k](gap_k))
            if radix > 1:
                x_k = [all_groups[r * cardinality + k] for r in range(radix)]
                x_k = torch.cat(x_k, dim=1)
                atten_k = atten_k.view(batch, radix, -1)
                atten_k = F.softmax(atten_k, dim=1)
            else:
                x_k = all_groups[k]
                atten_k = torch.sigmoid(atten_k)
            attended_k = x_k * atten_k.view(batch, -1, 1, 1)
            out_k = sum(torch.split(attended_k, attended_k.size(1)//self.radix, dim=1))
            out.append(out_k)
 
        return torch.cat(out, dim=1).contiguous()

@torch.no_grad()
def sync_weigths(m1, m2):
    m1.conv.weight.copy_(torch.from_numpy(m2.conv.weight.data.numpy()))
    nn.init.ones_(m1.fc1.weight)
    nn.init.ones_(m1.fc2.weight)
    nn.init.zeros_(m1.fc1.bias)
    nn.init.zeros_(m1.fc2.bias)
    for m in m2.fc1:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    for m in m2.fc2:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def _AssertTensorClose(a, b, atol=1e-3, rtol=1e-3):
    npa, npb = a.cpu().detach().numpy(), b.cpu().detach().numpy()
    assert np.allclose(npa, npb, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())

def test_radix_major():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    def compare_two_imp(batch, height, width,
                        in_channels, channels,
                        kernel_size, stride, padding,
                        radix, groups):
        layer1 = SplAtConv2d(in_channels, channels, kernel_size, stride, padding, radix=radix, groups=groups, bias=False)
        layer2 = RadixMajorNaiveImp(in_channels, channels, kernel_size, stride, padding, radix=radix, groups=groups, bias=False)
        sync_weigths(layer1, layer2)
        layer1 = layer1.to(device)
        layer2 = layer2.to(device)
        x = torch.rand(batch, in_channels, height, width).to(device)
        y1 = layer1(x)
        y2 = layer2(x)
        _AssertTensorClose(y1, y2)

    for batch in [2, 4, 8, 32]:
        for height in [7, 14, 28, 56]:
            width = height
            for in_channels in [16, 64, 128]:
                channels = in_channels
                for kernel_size in [3, 5]:
                     padding = kernel_size // 2
                     for stride in [1, 2]:
                        for radix in [1, 2, 4]:
                            for groups in [1, 2, 4]:
                                compare_two_imp(
                                    batch, height, width, in_channels,
                                    channels, kernel_size, stride, padding,
                                    radix, groups)

if __name__ == "__main__":
    import nose
    nose.runmodule()
