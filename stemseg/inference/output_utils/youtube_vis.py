from collections import defaultdict
from stemseg.config import cfg
from stemseg.inference.output_utils.common import annotate_instance
from stemseg.data.common import compute_resize_params_2
from stemseg.utils.timer import Timer
from pycocotools import mask as masktools
from stemseg.utils.vis import create_color_map
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import json
import numpy as np
import os
import torch
import torch.nn.functional as F


class YoutubeVISOutputGenerator(object):
    def __init__(self, output_dir, outlier_label, save_visualization, category_mapping, category_names, *args, **kwargs):
        # category_mapping contains a mapping from the original youtube vis class labels to the merged labels that were
        # used to train the model. This mapping now has to be reversed to obtain the original class labels for uploading
        # results.
        # self.category_mapping = defaultdict(list)
        # for orig_label, mapped_label in category_mapping.items():
        #     self.category_mapping[mapped_label].append(orig_label)  # one to many mapping is possible

        self.outlier_label = outlier_label
        self.instances = []

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.save_visualization = save_visualization

        self.category_names = category_names
        self.upscaled_inputs = kwargs.get("upscaled_inputs")

    @staticmethod
    def compute_instance_confidences(instance_pt_counts, instance_ids_to_keep):
        # set instance confidence based on number of points in the instance mask across the entire sequence.
        instance_pt_counts = {
            instance_id: count
            for instance_id, count in instance_pt_counts.items() if instance_id in instance_ids_to_keep
        }

        max_pts = float(max(list(instance_pt_counts.values())))
        return {
            instance_id: float(count) / max_pts for instance_id, count in instance_pt_counts.items()
        }

    @Timer.exclude_duration("postprocessing")
    def process_sequence(self, sequence, track_mask_idxes, track_mask_labels, instance_pt_counts, instance_lifetimes,
                         category_masks, mask_dims, mask_scale, max_tracks, device="cpu"):
        """
        Given a list of mask indices per frame, creates a sequence of masks for the entire sequence.
        :param sequence: instance of YoutubeVISSequence
        :param track_mask_idxes: list(tuple(tensor, tensor))
        :param track_mask_labels: list(tensor)
        :param instance_pt_counts: dict(int -> int)
        :param category_masks: tensor(T, C, H, W) of type float (result after softmax)
        :param mask_dims: tuple(int, int) (height, width)
        :param mask_scale: int
        :param max_tracks: int
        :param device: str
        :return: None
        """
        mask_height, mask_width = mask_dims
        image_height, image_width = sequence.image_dims
        assert len(track_mask_idxes) == len(track_mask_labels)

        if torch.is_tensor(category_masks):
            assert category_masks.shape[0] == len(track_mask_idxes)
            assert tuple(category_masks.shape[-2:]) == tuple(mask_dims), \
                "Shape mismatch between semantic masks {} and embedding masks {}".format(category_masks.shape, mask_dims)

        assert max_tracks < 256

        # filter out small/unstable instances
        instances_to_keep = [
            instance_id for instance_id, _ in sorted(
                [(k, v) for k, v in instance_lifetimes.items()], key=lambda x: x[1], reverse=True
            ) if instance_id != self.outlier_label
        ]
        instances_to_keep = instances_to_keep[:max_tracks]
        # num_tracks = len(instances_to_keep)
        print("Number of instances: ", len(instances_to_keep))
        if len(instances_to_keep) == 0:
            print("No instances detected for sequence {}".format(sequence.seq_id))
            return

        # move tensors to the target device
        track_mask_labels = [x.to(device=device) for x in track_mask_labels]
        track_mask_idxes = [(coords[0].to(device=device), coords[1].to(device=device)) for coords in track_mask_idxes]

        if torch.is_tensor(category_masks):
            category_masks = category_masks.permute(0, 2, 3, 1).to(device=device)  # [T, H, W, C]
        else:
            category_masks = [masks_per_frame.to(device=device) for masks_per_frame in category_masks]

        # instance_semantic_label_votes = defaultdict(lambda: {k: 0. for k in self.category_mapping.keys()})
        instance_semantic_label_logits = defaultdict(lambda: 0.)

        instance_rle_masks = {k: [] for k in instances_to_keep}
        instance_areas = {k: 0. for k in instances_to_keep}
        instance_confidences = self.compute_instance_confidences(instance_pt_counts, instances_to_keep)

        for t in range(len(track_mask_idxes)):
            # filter semantic labels for background pixels
            if torch.is_tensor(category_masks):
                category_mask_t = category_masks[t][track_mask_idxes[t]]
            else:
                category_mask_t = category_masks[t]

            assert category_mask_t.shape[0] == track_mask_labels[t].shape[0], \
                "Shape mismatch between category labels {} and instance labels {}".format(
                    category_mask_t.shape, track_mask_labels[t].shape)

            mask_t = []
            for instance_id in instances_to_keep:
                label_mask = track_mask_labels[t] == instance_id
                mask = torch.zeros(mask_height, mask_width, dtype=torch.long, device=device)
                mask[track_mask_idxes[t]] = label_mask.long()
                mask_t.append(mask)

                instance_category_preds = category_mask_t[label_mask].sum(dim=0)
                instance_areas[instance_id] += label_mask.sum(dtype=torch.float32)
                instance_semantic_label_logits[instance_id] += instance_category_preds[1:]  # remove background

            mask_t = torch.stack(mask_t, dim=0)

            # to obtain the mask in the original image dims:
            # 1. up-sample mask to network input size
            # 2. remove zero padding from right/bottom
            # 3. resize to original image dims

            mask_t = mask_t.unsqueeze(0).float()

            if not self.upscaled_inputs:
                mask_t = F.interpolate(mask_t, scale_factor=mask_scale, mode='bilinear', align_corners=False)

            # get resized network input dimensions (without zero padding)
            resized_mask_width, resized_mask_height, _ = compute_resize_params_2(
                (image_width, image_height), cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)

            try:
                assert mask_t.shape[3] >= resized_mask_width
                assert mask_t.shape[2] >= resized_mask_height
            except AssertionError as _:
                raise RuntimeError("Network input dims without padding {} should be <= padded dims".format(
                    (resized_mask_width, resized_mask_height), tuple(mask_t.shape[-2:])))

            # remove extra zeros
            mask_t = mask_t[:, :, :resized_mask_height, :resized_mask_width]

            # resize to original image dims
            mask_t = (F.interpolate(mask_t, (image_height, image_width), mode='bilinear', align_corners=False) > 0.5)
            mask_t = mask_t.byte().squeeze(0)

            for i, instance_id in enumerate(instances_to_keep):
                rle_mask = masktools.encode(np.asfortranarray(mask_t[i].cpu().numpy()))
                rle_mask["counts"] = rle_mask["counts"].decode("utf-8")  # bytes to utf-8 so that json.dump works
                instance_rle_masks[instance_id].append(rle_mask)

        self.add_sequence_result(sequence, instance_rle_masks, instance_semantic_label_logits, instance_areas, instance_confidences)

        return instances_to_keep, dict()

    def add_sequence_result(self, seq, instance_rle_masks, instance_semantic_label_logits, instance_areas, instance_confidences):
        # assign semantic label to each instance based on max votes
        # instance_mapped_labels = dict()
        instance_category_probs = dict()
        sequence_instances = []

        for instance_id in instance_rle_masks:
            instance_area = instance_areas[instance_id]
            semantic_label_probs = (instance_semantic_label_logits[instance_id] / instance_area).softmax(0).tolist()
            instance_category_probs[instance_id] = semantic_label_probs

            cat_id_probs_sorted = sorted([
                (cat_id, prob)
                for cat_id, prob in enumerate(semantic_label_probs, 1)
            ], key=lambda x: x[1], reverse=True)

            max_voted_label = cat_id_probs_sorted[0][0]

            sequence_instances.append({
                "video_id": seq.id,
                "score": instance_confidences[instance_id],
                "category_id": max_voted_label,
                "segmentations": instance_rle_masks[instance_id]
                # "category_probs": instance_category_probs[instance_id]
            })

        if self.save_visualization:
            self.save_sequence_visualizations(seq, sequence_instances)

        self.instances.extend(sequence_instances)

    @Timer.exclude_duration("postprocessing")
    def save_sequence_visualizations(self, seq, instances):
        cmap = create_color_map().tolist()
        seq_output_dir = os.path.join(self.output_dir, 'vis', str(seq.id))
        os.makedirs(seq_output_dir, exist_ok=True)

        images = seq.load_images()

        for t, image_t in enumerate(images):
            for n, instance in enumerate(instances, 1):
                segmentations = instance["segmentations"]
                assert len(segmentations) == len(images)

                category_label = self.category_names[instance["category_id"]]
                color = cmap[n]

                segmentation_t = segmentations[t].copy()
                segmentation_t["counts"] = segmentation_t["counts"].encode("utf-8")

                mask = masktools.decode(segmentation_t)

                annotation_text = "{} {:.2f}".format(category_label, instance["score"])
                image_t = annotate_instance(image_t, mask, color, annotation_text)

            cv2.imwrite(os.path.join(seq_output_dir, '{:05d}.jpg'.format(t)), image_t)

    def save(self, *args, **kwargs):
        """
        Writes out results to disk
        :return: None
        """
        # save results as JSON file
        output_json_path = os.path.join(self.output_dir, 'results.json')
        with open(output_json_path, 'w') as fh:
            json.dump(self.instances, fh)

        # zip the JSON file. For uploading to test server
        output_zip_path = os.path.join(self.output_dir, 'results.zip')
        zf = ZipFile(output_zip_path, 'w')
        zf.write(output_json_path, arcname='results.json')
        zf.close()

