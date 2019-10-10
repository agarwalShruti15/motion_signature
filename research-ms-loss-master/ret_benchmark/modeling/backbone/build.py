from ret_benchmark.modeling.registry import BACKBONES
from .vgg16 import VGG
from .bninception import BNInception


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME in BACKBONES, f"backbone {cfg.MODEL.BACKBONE} is not defined"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME]()