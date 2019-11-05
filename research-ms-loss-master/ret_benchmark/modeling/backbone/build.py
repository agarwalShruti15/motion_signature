from ret_benchmark.modeling.registry import BACKBONES
from .vgg16 import VGG
from .bninception import BNInception
from .alexnet import AlexNet
from .resnet50 import ResNet50
from .asym_resnet50 import AsymResNet50
from .asym_alexnet import AsymAlexNet
from .resnet18 import ResNet18
from .only_fc import Only_FC
from .only_fc_nodropout import Only_FC_noDP

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.NAME in BACKBONES, f"backbone {cfg.MODEL.BACKBONE} is not defined"
    return BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg)