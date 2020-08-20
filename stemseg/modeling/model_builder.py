from collections import OrderedDict
from functools import partial
from stemseg.utils import ModelOutputConsts as ModelOutput, ModelPaths, LossConsts
from stemseg.data.common import instance_masks_to_semseg_mask
from stemseg.modeling.losses import CrossEntropyLoss, EmbeddingLoss

from stemseg.utils.global_registry import GlobalRegistry

from stemseg.modeling.backbone import BACKBONE_REGISTRY
from stemseg.modeling.embedding_utils import get_nb_free_dims

from stemseg.modeling.embedding_decoder import EMBEDDING_HEAD_REGISTRY
from stemseg.modeling.semseg_decoder import SEMSEG_HEAD_REGISTRY
from stemseg.modeling.seediness_decoder import SEEDINESS_HEAD_REGISTRY

from stemseg.config import cfg

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


SEMSEG_LOSS_REGISTRY = GlobalRegistry.get("SemsegLoss")
SEMSEG_LOSS_REGISTRY.add("CrossEntropy", CrossEntropyLoss)

POOLER_REGISTRY = GlobalRegistry.get("PoolingLayer")
POOLER_REGISTRY.add("avg", nn.AvgPool3d)
POOLER_REGISTRY.add("max", nn.MaxPool3d)

NORM_REGISTRY = GlobalRegistry.get("NormalizationLayer")
NORM_REGISTRY.add("none", lambda num_groups: nn.Identity)
NORM_REGISTRY.add("gn", lambda num_groups: partial(nn.GroupNorm, num_groups))


class TrainingModel(nn.Module):
    def __init__(self, backbone, embedding_head, embedding_head_feature_map_scale, embedding_loss_criterion, semseg_head,
                 semseg_feature_map_scale, semseg_loss_criterion, seediness_head,
                 seediness_head_feature_map_scale, multiclass_semseg_output, output_resize_scale, logger):
        super(self.__class__, self).__init__()

        self.backbone = backbone

        # tracker related
        self.embedding_head = embedding_head
        self.embedding_head_feature_map_scale = embedding_head_feature_map_scale
        self.embedding_head_output_scale = min(self.embedding_head_feature_map_scale)

        self.embedding_loss_criterion = embedding_loss_criterion

        # semantic segmentation related
        self.semseg_head = semseg_head
        self.semseg_feature_map_scale = semseg_feature_map_scale
        self.semseg_output_scale = min(self.semseg_feature_map_scale)
        self.semseg_loss_criterion = semseg_loss_criterion

        # seediness head
        self.seediness_head = seediness_head
        self.seediness_head_feature_map_scale = seediness_head_feature_map_scale

        # feature map scale boiler plate
        all_feature_map_scales = self.embedding_head_feature_map_scale.copy()
        if self.semseg_head is not None:
            all_feature_map_scales += self.semseg_feature_map_scale

        min_scale_p = int(math.log2(min(all_feature_map_scales)))
        max_scale_p = int(math.log2(max(all_feature_map_scales)))
        self.feature_map_scales = [2**p for p in range(min_scale_p, max_scale_p + 1)]

        self.multiclass_semseg_output = multiclass_semseg_output
        self.output_resize_scale = output_resize_scale
        self.logger = logger

    def train(self, mode=True):
        self.training = mode
        for module_name, module in self.named_children():
            if module_name == "backbone" and cfg.TRAINING.FREEZE_BACKBONE:
                continue

            module.train(mode)
        return self

    def restore_temporal_dimension(self, x, num_seqs, num_frames, format):
        """
        Restores the temporal dimension given a flattened image/feature tensor
        :param x: tensor of shape [N*T, C, H, W]
        :param num_seqs: Number of image sequences (batch size)
        :param num_frames: Number of frames per image sequence
        :param format: Either 'NCTHW' or 'NTCHW'
        :return: tensor of shape defined by 'format' option
        """
        channels, height, width = x.shape[-3:]
        x = x.view(num_seqs, num_frames, channels, height, width)

        assert format in ["NCTHW", "NTCHW"]
        if format == "NCTHW":
            x = x.permute(0, 2, 1, 3, 4)
        return x

    def forward(self, image_seqs, targets):
        targets = self.resize_masks(targets)

        num_seqs = image_seqs.num_seqs
        num_frames = image_seqs.num_frames
        features = self.run_backbone(image_seqs)

        embeddings_map, semseg_logits = self.forward_embeddings_and_semseg(features, num_seqs, num_frames)

        output = {
            ModelOutput.INFERENCE: {
                ModelOutput.EMBEDDINGS: embeddings_map,
                ModelOutput.SEMSEG_MASKS: semseg_logits,
            },
        }

        self.embedding_loss_criterion(embeddings_map, targets, output)

        if self.semseg_head is not None:
            if self.semseg_head.has_foreground_channel:
                semseg_logits, fg_logits = semseg_logits.split((semseg_logits.shape[2] - 1, 1), dim=2)
                self.compute_fg_loss(fg_logits.squeeze(2), targets, output)

            self.semseg_loss_criterion(semseg_logits, targets, output)

        return output

    @torch.no_grad()
    def resize_masks(self, targets):
        """
        Downscales masks to the required size
        :param targets:
        :return: dict
        """
        assert self.embedding_head_output_scale == self.semseg_output_scale

        for target in targets:
            if self.output_resize_scale == 1.0:
                target['masks'] = F.interpolate(target['masks'].float(),
                                                scale_factor=1./self.embedding_head_output_scale,
                                                mode='bilinear', align_corners=False)
                target['masks'] = target['masks'].byte().detach()

                target['ignore_masks'] = F.interpolate(target['ignore_masks'].unsqueeze(0).float(),
                                                       scale_factor=1. / self.semseg_output_scale,
                                                       mode='bilinear', align_corners=False)
                target['ignore_masks'] = target['ignore_masks'].squeeze(0).byte().detach()

            if self.semseg_head is not None:
                target['semseg_masks'] = instance_masks_to_semseg_mask(target['masks'], target['category_ids'])

        return targets

    def run_backbone(self, image_seqs):
        """
        Computes backbone features for a set of image sequences.
        :param image_seqs: Instance of ImageList
        :return: A dictionary of feature maps with keys denoting the scale.
        """
        height, width = image_seqs.tensors.shape[-2:]
        images_tensor = image_seqs.tensors.view(image_seqs.num_seqs * image_seqs.num_frames, 3, height, width)

        if cfg.TRAINING.FREEZE_BACKBONE:
            with torch.no_grad():
                features = self.backbone(images_tensor)
        else:
            features = self.backbone(images_tensor)

        return OrderedDict([(k, v) for k, v in zip(self.feature_map_scales, features)])

    def forward_embeddings_and_semseg(self, features, num_seqs, num_frames):
        if self.semseg_head is None:
            semseg_logits = None
        else:
            semseg_input_features = [
                self.restore_temporal_dimension(features[scale], num_seqs, num_frames, "NCTHW")
                for scale in self.semseg_feature_map_scale
            ]
            semseg_logits = self.semseg_head(semseg_input_features)  # [N, C, T, H, W]
            semseg_logits = semseg_logits.permute(0, 2, 1, 3, 4)  # [N, T, C, H, W]

        embedding_head_input_features = [
            self.restore_temporal_dimension(features[scale], num_seqs, num_frames, "NCTHW")
            for scale in self.embedding_head_feature_map_scale
        ]
        embeddings_map = self.embedding_head(embedding_head_input_features)

        if self.seediness_head is not None:
            seediness_input_features = [
                self.restore_temporal_dimension(features[scale], num_seqs, num_frames, "NCTHW")
                for scale in self.seediness_head_feature_map_scale
            ]
            seediness_map = self.seediness_head(seediness_input_features)

            embeddings_map = torch.cat((embeddings_map, seediness_map), dim=1)

        if self.output_resize_scale != 1.0:
            embeddings_map = F.interpolate(
                embeddings_map, scale_factor=(1.0, self.output_resize_scale, self.output_resize_scale),
                mode='trilinear', align_corners=False
            )
            if torch.is_tensor(semseg_logits):
                semseg_logits = F.interpolate(
                    semseg_logits, scale_factor=(1.0, self.output_resize_scale, self.output_resize_scale),
                    mode='trilinear', align_corners=False
                )

        return embeddings_map, semseg_logits

    def compute_fg_loss(self, fg_logits, targets, output_dict):
        """
        Computes the foreground/background loss
        :param fg_logits: tensor(N, T, H, W)
        :param targets: dict
        :param output_dict: dict
        :return: loss
        """
        loss = 0.

        for pred_fg_logits_per_seq, targets_per_seq in zip(fg_logits, targets):
            gt_semseg_masks_per_seq = targets_per_seq['semseg_masks']
            ignore_masks_per_seq = targets_per_seq['ignore_masks']

            assert gt_semseg_masks_per_seq.shape[-2:] == pred_fg_logits_per_seq.shape[-2:], \
                "Shape mismatch between ground truth semseg masks {} and predicted semseg masks {}".format(
                    gt_semseg_masks_per_seq.shape, pred_fg_logits_per_seq.shape
                )
            assert gt_semseg_masks_per_seq.shape[-2:] == ignore_masks_per_seq.shape[-2:], \
                "Shape mismatch between ground truth semseg masks {} and ignore masks {} ".format(
                    gt_semseg_masks_per_seq.shape, ignore_masks_per_seq.shape
                )

            fg_masks_per_seq = (gt_semseg_masks_per_seq > 0).float()
            seq_loss = F.binary_cross_entropy_with_logits(pred_fg_logits_per_seq, fg_masks_per_seq, reduction="none")

            with torch.no_grad():
                nonignore_masks_per_seq = 1. - ignore_masks_per_seq.float()

            seq_loss = seq_loss * nonignore_masks_per_seq
            seq_loss = seq_loss.sum() / nonignore_masks_per_seq.sum().detach()

            loss = loss + seq_loss

        output_dict[ModelOutput.OPTIMIZATION_LOSSES][LossConsts.FOREGROUND] = loss / len(targets)


def build_model(restore_pretrained_backbone_wts=False, logger=None):
    print_fn = logger.info if logger is not None else print

    # manually seed the random number generator so that all weights get initialized to the same values when using
    # torch.nn.parallel.DistributedDataParallel
    torch.manual_seed(42)

    # build backbone network
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    backbone_builder = BACKBONE_REGISTRY[backbone_type]
    backbone = backbone_builder(cfg)

    info_to_print = [
        "Backbone type: {}".format(cfg.MODEL.BACKBONE.TYPE),
        "Backbone frozen: {}".format("Yes" if cfg.TRAINING.FREEZE_BACKBONE else "No")
    ]

    # restore pre-trained weights if possible.
    if restore_pretrained_backbone_wts:
        pretrained_wts_file = os.path.join(ModelPaths.pretrained_backbones_dir(), cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS)
        print_fn("Restoring backbone weights from '{}'".format(pretrained_wts_file))
        if os.path.exists(pretrained_wts_file):
            restore_dict = torch.load(pretrained_wts_file)
            backbone.load_state_dict(restore_dict, strict=True)
        else:
            raise ValueError("Could not find pre-trained backbone weights file at expected location: '{}'".format(
                pretrained_wts_file))

    embedding_head_seediness_output = not cfg.MODEL.USE_SEEDINESS_HEAD

    add_semseg_head = cfg.MODEL.USE_SEMSEG_HEAD
    if cfg.INPUT.NUM_CLASSES > 2:
        assert add_semseg_head, "Number of object classes > 2, but 'USE_SEMSEG_HEAD' option is set to False"

    # create embedding/tracker head
    EmbeddingHeadType = EMBEDDING_HEAD_REGISTRY[cfg.MODEL.EMBEDDINGS.HEAD_TYPE]

    embedding_head = EmbeddingHeadType(
        backbone.out_channels, cfg.MODEL.EMBEDDINGS.INTER_CHANNELS, cfg.MODEL.EMBEDDINGS.EMBEDDING_SIZE,
        tanh_activation=cfg.MODEL.EMBEDDINGS.TANH_ACTIVATION,
        seediness_output=embedding_head_seediness_output,
        experimental_dims=cfg.MODEL.EMBEDDING_DIM_MODE,
        PoolType=POOLER_REGISTRY[cfg.MODEL.EMBEDDINGS.POOL_TYPE],
        NormType=NORM_REGISTRY[cfg.MODEL.EMBEDDINGS.NORMALIZATION_LAYER](cfg.MODEL.EMBEDDINGS.GN_NUM_GROUPS),
    )

    # create embedding loss criterion
    embedding_loss_criterion = EmbeddingLoss(
        min(cfg.MODEL.EMBEDDINGS.SCALE),
        embedding_size=cfg.MODEL.EMBEDDINGS.EMBEDDING_SIZE,
        nbr_free_dims=get_nb_free_dims(cfg.MODEL.EMBEDDING_DIM_MODE),
        **cfg.TRAINING.LOSSES.EMBEDDING.d())

    info_to_print.append("Embedding head type: {}".format(cfg.MODEL.EMBEDDINGS.HEAD_TYPE))
    info_to_print.append("Embedding head channels: {}".format(cfg.MODEL.EMBEDDINGS.INTER_CHANNELS))
    info_to_print.append("Embedding dims: {}".format(cfg.MODEL.EMBEDDINGS.EMBEDDING_SIZE))
    info_to_print.append("Embedding dim mode: {}".format(cfg.MODEL.EMBEDDING_DIM_MODE))
    info_to_print.append("Embedding free dim stds: {}".format(cfg.TRAINING.LOSSES.EMBEDDING.FREE_DIM_STDS))
    info_to_print.append("Embedding head normalization: {}".format(cfg.MODEL.EMBEDDINGS.NORMALIZATION_LAYER))
    info_to_print.append("Embedding head pooling type: {}".format(cfg.MODEL.EMBEDDINGS.POOL_TYPE))

    if cfg.MODEL.USE_SEEDINESS_HEAD:
        SeedinessHeadType = SEEDINESS_HEAD_REGISTRY[cfg.MODEL.SEEDINESS.HEAD_TYPE]
        seediness_head = SeedinessHeadType(
            backbone.out_channels, cfg.MODEL.SEEDINESS.INTER_CHANNELS,
            PoolType=POOLER_REGISTRY[cfg.MODEL.SEEDINESS.POOL_TYPE],
            NormType=NORM_REGISTRY[cfg.MODEL.SEEDINESS.NORMALIZATION_LAYER](cfg.MODEL.SEEDINESS.GN_NUM_GROUPS)
        )
        info_to_print.append("Seediness head type: {}".format(cfg.MODEL.SEEDINESS.HEAD_TYPE))
        info_to_print.append("Seediness head channels: {}".format(cfg.MODEL.SEEDINESS.INTER_CHANNELS))
        info_to_print.append("Seediness head normalization: {}".format(cfg.MODEL.SEEDINESS.NORMALIZATION_LAYER))
        info_to_print.append("Seediness head pooling type: {}".format(cfg.MODEL.SEEDINESS.POOL_TYPE))
    else:
        seediness_head = None
        info_to_print.append("Seediness head type: N/A")

    # create semantic segmentation head
    if add_semseg_head:
        SemsegHeadType = SEMSEG_HEAD_REGISTRY[cfg.MODEL.SEMSEG.HEAD_TYPE]
        semseg_head = SemsegHeadType(
            backbone.out_channels, cfg.INPUT.NUM_CLASSES,
            inter_channels=cfg.MODEL.SEMSEG.INTER_CHANNELS, feature_scales=cfg.MODEL.SEMSEG.FEATURE_SCALE,
            foreground_channel=cfg.MODEL.SEMSEG.FOREGROUND_CHANNEL, PoolType=POOLER_REGISTRY[cfg.MODEL.SEMSEG.POOL_TYPE],
            NormType=NORM_REGISTRY[cfg.MODEL.SEMSEG.NORMALIZATION_LAYER](cfg.MODEL.SEMSEG.GN_NUM_GROUPS)
        )

        # create semseg loss criterion
        SemsegLossType = SEMSEG_LOSS_REGISTRY[cfg.TRAINING.LOSSES.SEMSEG]
        semseg_loss_criterion = SemsegLossType()

        info_to_print.append("Semseg head type: {}".format(cfg.MODEL.SEMSEG.HEAD_TYPE))
        info_to_print.append("Semseg head channels: {}".format(cfg.MODEL.SEMSEG.INTER_CHANNELS))
        info_to_print.append("Sesmeg with foreground channel: {}".format("Yes" if cfg.MODEL.SEMSEG.FOREGROUND_CHANNEL else "No"))
        info_to_print.append("Semseg loss type: {}".format(cfg.TRAINING.LOSSES.SEMSEG))
        info_to_print.append("Semseg head normalization: {}".format(cfg.MODEL.SEMSEG.NORMALIZATION_LAYER))
        info_to_print.append("Semseg head pooling type: {}".format(cfg.MODEL.SEMSEG.POOL_TYPE))
    else:
        semseg_head = None
        semseg_loss_criterion = None
        info_to_print.append("Semseg head type: N/A")

    multiclass_semseg_output = cfg.INPUT.NUM_CLASSES > 2
    output_resize_scale = 4.0 if cfg.TRAINING.LOSS_AT_FULL_RES else 1.0

    info_to_print.append("Output resize scale: {}".format(output_resize_scale))

    print_fn("Model configuration\n"
             "{}\n".format("\n".join(["  - {}".format(line) for line in info_to_print])))

    return TrainingModel(
        backbone=backbone,
        embedding_head=embedding_head,
        embedding_head_feature_map_scale=cfg.MODEL.EMBEDDINGS.SCALE,
        embedding_loss_criterion=embedding_loss_criterion,
        semseg_head=semseg_head,
        semseg_feature_map_scale=cfg.MODEL.SEMSEG.FEATURE_SCALE,
        semseg_loss_criterion=semseg_loss_criterion,
        seediness_head=seediness_head,
        seediness_head_feature_map_scale=cfg.MODEL.SEEDINESS.FEATURE_SCALE,
        multiclass_semseg_output=multiclass_semseg_output,
        output_resize_scale=output_resize_scale,
        logger=logger
    )
