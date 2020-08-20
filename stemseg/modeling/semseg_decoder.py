import torch
import torch.nn as nn

from stemseg.modeling.common import UpsampleTrilinear3D, AtrousPyramid3D, get_pooling_layer_creator, \
    get_temporal_scales
from stemseg.utils.global_registry import GlobalRegistry


SEMSEG_HEAD_REGISTRY = GlobalRegistry.get("SemsegHead")


@SEMSEG_HEAD_REGISTRY.add("squeeze_expand_decoder")
class SqueezeExpandDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, inter_channels, feature_scales, foreground_channel=False,
                 ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()
        self.is_3d = True

        assert tuple(feature_scales) == (4, 8, 16, 32)

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        self.block_32x = nn.Sequential(
            ConvType(in_channels, inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2](3, stride=(2, 1, 1), padding=1),
        )

        self.block_16x = nn.Sequential(
            ConvType(in_channels, inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[1], inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
            # ResidualModuleWrapper(NonLocalBlock3DWithDownsamplingV2(inter_channels, 128, 1))
        )

        self.block_8x = nn.Sequential(
            ConvType(in_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(in_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        out_channels = num_classes + 1 if foreground_channel else num_classes
        self.conv_out = nn.Conv3d(inter_channels[3], out_channels, kernel_size=1, padding=0, bias=False)

        self.has_foreground_channel = foreground_channel

    def forward(self, x):
        assert len(x) == 4, "Expected 4 feature maps, got {}".format(len(x))

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x[::-1]

        feat_map_32x = self.block_32x(feat_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(feat_map_32x)
        feat_map_16x = self.block_16x(feat_map_16x)
        x = torch.cat((x, feat_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        feat_map_8x = self.block_8x(feat_map_8x)
        x = torch.cat((x, feat_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        feat_map_4x = self.block_4x(feat_map_4x)
        x = torch.cat((x, feat_map_4x), 1)
        x = self.conv_4(x)

        return self.conv_out(x)


class SqueezeExpandDilatedDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, inter_channels, feature_scales, foreground_channel=False,
                 ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()

        assert tuple(feature_scales) == (4, 8, 16, 32)

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        self.block_32x = nn.Sequential(
            AtrousPyramid3D(in_channels, 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(inter_channels[0], 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(inter_channels[0], 64, ((1, 3, 3), (1, 6, 6), (1, 9, 9)), inter_channels[0]),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        )

        self.block_16x = nn.Sequential(
            AtrousPyramid3D(in_channels, 64, ((1, 4, 4), (1, 8, 8), (1, 12, 12)), inter_channels[1]),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),

            AtrousPyramid3D(in_channels, 64, ((1, 4, 4), (1, 8, 8), (1, 12, 12)), inter_channels[1]),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1]((3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        )

        self.block_8x = nn.Sequential(
            ConvType(in_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(in_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        # output layer
        out_channels = num_classes + 1 if foreground_channel else num_classes
        self.conv_out = nn.Conv3d(inter_channels[-1], out_channels, kernel_size=1, padding=0, bias=False)

        self.has_foreground_channel = foreground_channel

    def forward(self, x):
        assert len(x) == 4, "Expected 4 feature maps, got {}".format(len(x))

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x[::-1]

        feat_map_32x = self.block_32x(feat_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(feat_map_32x)
        feat_map_16x = self.block_16x(feat_map_16x)
        x = torch.cat((x, feat_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        feat_map_8x = self.block_8x(feat_map_8x)
        x = torch.cat((x, feat_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        feat_map_4x = self.block_4x(feat_map_4x)
        x = torch.cat((x, feat_map_4x), 1)
        x = self.conv_4(x)

        return self.conv_out(x)
