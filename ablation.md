## Pretrained Models

|                 | setting | #P    | GFLOPs | PyTorch | Gluon |
|-----------------|---------|-------|--------|---------|-------|
| ResNeSt-50-fast | 1s1x64d | 26.3M | 4.34   | 80.33   | 80.35 |
| ResNeSt-50-fast | 2s1x64d | 27.5M | 4.34   | 80.53   | 80.65 |
| ResNeSt-50-fast | 4s1x64d | 31.9M | 4.35   | 80.76   | 80.90 |
| ResNeSt-50-fast | 1s2x40d | 25.9M | 4.38   | 80.59   | 80.72 |
| ResNeSt-50-fast | 2s2x40d | 26.9M | 4.38   | 80.61   | 80.84 |
| ResNeSt-50-fast | 4s2x40d | 30.4M | 4.41   | 81.14   | 81.17 |
| ResNeSt-50-fast | 1s4x24d | 25.7M | 4.42   | 80.99   | 80.97 |

### PyTorch Models

- Load using Torch Hub

```python
import torch
# get list of models
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

# load pretrained models, using ResNeSt-50-fast_2s1x64d as an example
net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_2s1x64d', pretrained=True)
```


- Load using python package

```python
# using ResNeSt-50 as an example
from resnest.torch import resnest50_fast_2s1x64d
net = resnest50_fast_2s1x64d(pretrained=True)
```


### Gluon Models

- Load pretrained model:

```python
# using ResNeSt-50 as an example
from resnest.gluon import resnest50_fast_2s1x64d
net = resnest50_fast_2s1x64d(pretrained=True)
```

