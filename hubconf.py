dependencies = ['torch']

from resnest.torch import resnest50, resnest101, resnest200, resnest269
from resnest.torch import (resnest50_fast_1s1x64d, resnest50_fast_2s1x64d, resnest50_fast_4s1x64d,
                                    resnest50_fast_1s2x40d, resnest50_fast_2s2x40d, resnest50_fast_4s2x40d,
                                    resnest50_fast_1s4x24d)
