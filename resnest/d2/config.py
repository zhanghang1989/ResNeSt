from detectron2.config import CfgNode as CN

def add_resnest_config(cfg):
    """Add config for ResNeSt
    """
    cfg.MODEL.RESNETS.RADIX = 2
    cfg.MODEL.RESNETS.DEEP_STEM = True
    cfg.MODEL.RESNETS.AVD = True
    cfg.MODEL.RESNETS.AVG_DOWN = True
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64
