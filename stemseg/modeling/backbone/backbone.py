# This script has been reproduced with slight modification from Facebook AI's maskrcnn-benchmark repository at:
# https://github.com/facebookresearch/maskrcnn-benchmark

from collections import OrderedDict

from torch import nn

from stemseg.modeling.backbone.make_layers import conv_with_kaiming_uniform
from stemseg.modeling.backbone import fpn as fpn_module
from stemseg.modeling.backbone import resnet


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    model.is_3d = False
    return model
