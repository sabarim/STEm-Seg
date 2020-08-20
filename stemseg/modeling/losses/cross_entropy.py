from stemseg.utils import ModelOutputConsts, LossConsts
from stemseg.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, semseg_logits, targets, output_dict):
        """
        Computes the semantic segmentation loss
        :param semseg_logits: tensor of shape [N, T, cls, H, W]
        :param targets: list(dict(tensors))
        :return: scalar loss for semantic segmentation
        """
        loss = 0.

        for pred_semseg_logits_per_seq, targets_per_seq in zip(semseg_logits, targets):
            gt_semseg_masks_per_seq = targets_per_seq['semseg_masks']
            ignore_masks_per_seq = targets_per_seq['ignore_masks']

            assert gt_semseg_masks_per_seq.shape[-2:] == pred_semseg_logits_per_seq.shape[-2:], \
                "Shape mismatch between ground truth semseg masks {} and predicted semseg masks {}".format(
                    gt_semseg_masks_per_seq.shape, pred_semseg_logits_per_seq.shape
                )
            assert gt_semseg_masks_per_seq.shape[-2:] == ignore_masks_per_seq.shape[-2:], \
                "Shape mismatch between ground truth semseg masks {} and ignore masks {} ".format(
                    gt_semseg_masks_per_seq.shape, ignore_masks_per_seq.shape
                )

            seq_loss = F.cross_entropy(pred_semseg_logits_per_seq, gt_semseg_masks_per_seq)

            with torch.no_grad():
                nonignore_masks_per_seq = 1. - ignore_masks_per_seq.float()

            seq_loss = seq_loss * nonignore_masks_per_seq
            seq_loss = seq_loss.sum() / nonignore_masks_per_seq.sum().detach()

            loss = loss + seq_loss

        loss = loss / len(targets)

        output_dict[ModelOutputConsts.OTHERS][LossConsts.SEMSEG] = loss
        output_dict[ModelOutputConsts.OPTIMIZATION_LOSSES][LossConsts.SEMSEG] = loss * cfg.TRAINING.LOSSES.WEIGHT_SEMSEG
