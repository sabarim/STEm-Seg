from collections import defaultdict, namedtuple
from stemseg.data import InferenceImageLoader
from stemseg.data.inference_image_loader import collate_fn
from stemseg.modeling.model_builder import build_model
from stemseg.utils.timer import Timer

from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceModel(nn.Module):
    def __init__(self, restore_path=None, cpu_workers=4, preload_images=False, semseg_output_type="probs",
                 resize_scale=1.0, semseg_generation_on_gpu=True):
        super().__init__()

        with torch.no_grad():
            self._model = build_model(restore_pretrained_backbone_wts=False)

        if self._model.backbone.is_3d:
            raise ValueError("Only implemend for 2D backbones")

        if restore_path:
            self._model.load_state_dict(torch.load(restore_path)['model'])

        self.cpu_workers = cpu_workers

        self.EmbeddingMapEntry = namedtuple(
            "EmbeddingMapEntry", ["subseq_frames", "embeddings", "bandwidths", "seediness"])

        self.preload_images = preload_images

        self.semseg_output_type = semseg_output_type
        self.resize_scale = resize_scale
        self.semseg_generation_on_gpu = semseg_generation_on_gpu

        self.eval()

    @property
    def mask_scale(self):
        return self._model.semseg_output_scale

    @property
    def has_semseg_head(self):
        return self._model.semseg_head is not None

    @Timer.exclude_duration("inference", "postprocessing")
    def load_images(self, image_paths):
        return [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]

    @torch.no_grad()
    def resize_output(self, x):
        if self.resize_scale != 1.0:
            return F.interpolate(x, scale_factor=(1.0, self.resize_scale, self.resize_scale), mode='trilinear',
                                 align_corners=False)
        else:
            return x

    @torch.no_grad()
    def forward(self, image_paths, subseq_idxes):
        """
        Initialize a new sequence of images (arbitrary length)
        :param image_paths: list of file paths to the images
        :param subseq_idxes: list of tuples containing frame indices of the sub-sequences
        """

        # create an image loader
        if self.preload_images:
            image_loader = InferenceImageLoader(self.load_images(image_paths))
        else:
            image_loader = InferenceImageLoader(image_paths)

        image_loader = DataLoader(image_loader, 1, False, num_workers=self.cpu_workers, collate_fn=collate_fn,
                                  drop_last=False)

        semseg_logits = [[0., 0] for _ in range(len(image_paths))]
        embeddings_maps = []

        backbone_features = dict()
        current_subseq_idx = 0

        # to avoid recomputing features for the same frame again and again, we construct a dict to store the subseq
        # indices which are dependent on each frame.
        subseq_deps = defaultdict(set)
        for i, subseq in enumerate(subseq_idxes):
            for t in subseq:
                subseq_deps[t].add(i)

        # sub-sequences are allowed to have duplicate indices. This is useful when, for example, an entire sequence is
        # smaller than the network's expected temporal input and the first image is repeated. In this case, we
        # want to avoid running the backbone multiple times for the same image.
        current_subseq = {t: False for t in subseq_idxes[0]}
        current_subseq_as_list = subseq_idxes[0]

        for images, idxes in tqdm(image_loader, total=len(image_loader)):
            assert len(idxes) == 1
            frame_id = idxes[0]
            backbone_features[frame_id] = self._model.run_backbone(images.cuda())

            if frame_id in current_subseq:
                current_subseq[frame_id] = True

            if not all(list(current_subseq.values())):
                continue

            # required feature maps have been generated. Stack the feature maps and run semseg, embedding and seediness
            # heads
            stacked_features = defaultdict(list)
            for t in current_subseq_as_list:
                for scale, feature_map in backbone_features[t].items():
                    stacked_features[scale].append(feature_map)

            stacked_features = {
                scale: torch.stack(stacked_features[scale], 2) for scale in stacked_features
            }  # dict(tensor(1, C, T, H, W))

            if self.has_semseg_head:
                semseg_input_features = [stacked_features[scale] for scale in self._model.semseg_feature_map_scale]
                subseq_semseg_logits = self._model.semseg_head(semseg_input_features)
                subseq_semseg_logits = self.resize_output(subseq_semseg_logits).permute(2, 0, 1, 3, 4).cpu()

                for i, t in enumerate(current_subseq_as_list):
                    semseg_logits[t][0] += subseq_semseg_logits[i]
                    semseg_logits[t][1] += 1

            embedding_input_features = [stacked_features[scale] for scale in self._model.embedding_head_feature_map_scale]

            embedding_head_output = self._model.embedding_head(embedding_input_features).squeeze(0)

            # embedding_head_output = self._model.embedding_head(embedding_input_features)
            # embedding_head_output = self.resize_output(embedding_head_output).squeeze(0)

            embedding_head_output_dict = {t: embedding_head_output[:, i] for i, t in enumerate(current_subseq_as_list)}
            embedding_head_output = torch.stack([embedding_head_output_dict[t] for t in sorted(current_subseq.keys())], 1)

            subseq_embeddings, subseq_bandwidths, subseq_seediness = embedding_head_output.split(
                (
                    self._model.embedding_head.embedding_size,
                    self._model.embedding_head.variance_channels,
                    self._model.embedding_head.seediness_channels
                ), dim=0
            )

            subseq_bandwidths = subseq_bandwidths.exp() * 10.

            if subseq_seediness.numel() == 0:
                assert self._model.seediness_head is not None
                seediness_input_features = [
                    stacked_features[scale] for scale in self._model.seediness_head_feature_map_scale
                ]
                subseq_seediness = self._model.seediness_head(seediness_input_features)
                subseq_seediness = self.resize_output(subseq_seediness).squeeze(0)

                subseq_seediness_dict = {t: subseq_seediness[:, i] for i, t in enumerate(current_subseq_as_list)}
                subseq_seediness = torch.stack([subseq_seediness_dict[t] for t in sorted(current_subseq.keys())], 1)

            embeddings_maps.append(self.EmbeddingMapEntry(
                sorted(current_subseq.keys()), subseq_embeddings.cpu(), subseq_bandwidths.cpu(), subseq_seediness.cpu()))

            # clear backbone feature maps which are not needed for the next sub-sequence
            frames_to_discard = set()
            for frame_id, subseqs in subseq_deps.items():
                subseqs.discard(current_subseq_idx)
                if len(subseqs) == 0:
                    frames_to_discard.add(frame_id)

            backbone_features = {
                t: feature_map for t, feature_map in backbone_features.items() if t not in frames_to_discard
            }

            # update current sub-sequence
            current_subseq_idx += 1
            if current_subseq_idx == len(subseq_idxes):
                continue

            current_subseq = {idx: False for idx in subseq_idxes[current_subseq_idx]}
            current_subseq_as_list = subseq_idxes[current_subseq_idx]

            for t in backbone_features:
                if t in current_subseq:
                    current_subseq[t] = True

        # compute semseg probabilities
        fg_masks, multiclass_masks = self.get_semseg_masks(semseg_logits)

        return {
            "fg_masks": fg_masks,
            "multiclass_masks": multiclass_masks,
            "embeddings": embeddings_maps
        }

    @torch.no_grad()
    def get_semseg_masks(self, semseg_logits):
        """
        :param semseg_logits: list(tuple(tensor, int))
        :return: tensor(T, C, H, W) or tensor(T, H, W)
        """
        fg_masks, multiclass_masks = [], []
        if self._model.semseg_head is None:
            return fg_masks, multiclass_masks

        device = "cuda:0" if self.semseg_generation_on_gpu else "cpu"
        semseg_logits = torch.cat([(logits.to(device=device) / float(num_entries)) for logits, num_entries in semseg_logits], 0)

        if semseg_logits.shape[1] > 2:
            # multi-class segmentation: first N-1 channels correspond to logits for N-1 classes and the Nth channels is
            # a fg/bg mask
            multiclass_logits, fg_logits = semseg_logits.split((semseg_logits.shape[1] - 1, 1), dim=1)

            if self.semseg_output_type == "logits":
                multiclass_masks.append(multiclass_logits)
            elif self.semseg_output_type == "probs":
                multiclass_masks.append(F.softmax(multiclass_logits, dim=1))
            elif self.semseg_output_type == "argmax":
                multiclass_masks.append(multiclass_logits.argmax(dim=1))

            fg_masks.append(fg_logits.squeeze(1).sigmoid())

        else:
            # only fg/bg segmentation: the 2 channels correspond to bg and fg logits, respectively
            fg_masks.append(F.softmax(semseg_logits, dim=1)[:, 1])

        fg_masks = torch.cat(fg_masks)
        if multiclass_masks:
            multiclass_masks = torch.cat(multiclass_masks)

        return fg_masks.cpu(), multiclass_masks.cpu()
