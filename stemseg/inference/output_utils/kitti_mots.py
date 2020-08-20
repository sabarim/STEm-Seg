from collections import defaultdict
from stemseg.config import cfg
from stemseg.data.common import compute_resize_params_2
from stemseg.inference.output_utils import annotate_instance
from stemseg.inference.output_utils.kitti_mots_postprocessing import main as postprocess_results
from stemseg.utils.timer import Timer
from stemseg.utils.vis import create_color_map, overlay_mask_on_image

from tqdm import tqdm
import cv2
import numpy as np
import pycocotools.mask as masktools
import os
import torch
import torch.nn.functional as F


class KittiMOTSOutputGenerator(object):
    def __init__(self, output_dir, outlier_label, save_visualization, *args, **kwargs):
        self.results_output_dir = os.path.join(output_dir, "results")
        self.vis_output_dir = os.path.join(output_dir, "vis")

        self.outlier_label = outlier_label
        self.save_visualization = save_visualization

        self.categories = (1, 2)
        self.category_label = {1: "c", 2: "p"}  # car, pedestrian
        self.upscaled_inputs = kwargs.get("upscaled_inputs")

    @Timer.exclude_duration("postprocessing")
    def process_sequence(self, sequence, track_mask_idxes, track_mask_labels, instance_pt_counts, instance_lifetimes,
                         category_masks, mask_dims, mask_scale, max_tracks, device="cpu"):
        """
        Given a list of mask indices per frame, creates a sequence of masks for the entire sequence.
        :param sequence: instance of MOTSSequence
        :param track_mask_idxes: list(tuple(tensor, tensor))
        :param track_mask_labels: list(tensor)
        :param instance_pt_counts: dict(int -> int)
        :param instance_lifetimes: dict(int -> int)
        :param category_masks: tensor(T, H, W) of type long
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

        # filter out small/unstable instances
        instances_to_keep = [
            instance_id for instance_id, _ in sorted(
                [(k, v) for k, v in instance_pt_counts.items()], key=lambda x: x[1], reverse=True
            ) if instance_id != self.outlier_label
        ]
        instances_to_keep = instances_to_keep[:max_tracks]

        # MOTS eval does not allow the same pixel to be assigned to multiple instances. To break ties, we assign a
        # pixel to the instance with longer life-time in case of conflicts.

        # reorder the instances in descending order of temporal lifetime
        instances_to_keep = sorted(instances_to_keep, key=lambda x: instance_lifetimes[x])

        # map the instance IDs to a the range [1, N]
        instance_id_mapping = {instance_id: i for i, instance_id in enumerate(instances_to_keep, 1)}

        if len(instances_to_keep) == 0:
            raise ValueError("Zero instances detected in sequence: {}".format(sequence.id))

        instance_semantic_label_votes = defaultdict(lambda: {cat_id: 0 for cat_id in self.categories})
        instance_rle_masks = {k: [] for k in instance_id_mapping.values()}

        # move tensors to the target device
        track_mask_labels = [x.to(device=device) for x in track_mask_labels]
        track_mask_idxes = [(coords[0].to(device=device), coords[1].to(device=device)) for coords in track_mask_idxes]

        if torch.is_tensor(category_masks):
            category_masks = category_masks.to(device=device)
        else:
            category_masks = [masks_per_frame.to(device=device) for masks_per_frame in category_masks]

        print("Producing mask outputs...")
        for t in tqdm(range(len(track_mask_idxes))):
            if torch.is_tensor(category_masks):
                # filter semantic labels for background pixels
                category_mask_t = category_masks[t][track_mask_idxes[t]]
            else:
                category_mask_t = category_masks[t]

            assert category_mask_t.shape == track_mask_labels[t].shape, \
                "Shape mismatch between category labels {} and instance labels {}".format(
                    category_mask_t.shape, track_mask_labels[t].shape)

            mask_t = []
            active_instances_t = []
            for i, instance_id in enumerate(instances_to_keep, 1):
                label_mask = track_mask_labels[t] == instance_id
                if label_mask.sum(dtype=torch.long) == 0:
                    continue

                active_instances_t.append(instance_id)
                mask = torch.zeros(mask_height, mask_width, dtype=torch.long, device=device)
                mask[track_mask_idxes[t]] = label_mask.long()
                mask_t.append(mask)

                # count votes for the semantic label of each instance
                active_semantic_labels, label_counts = category_mask_t[label_mask].unique(return_counts=True)

                active_semantic_labels = active_semantic_labels.tolist()
                label_counts = label_counts.tolist()

                for label, count in zip(active_semantic_labels, label_counts):
                    if label != 0:
                        instance_semantic_label_votes[instance_id_mapping[instance_id]][label] += count

            if not mask_t:  # no instances in frame 't'
                continue

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
                raise RuntimeError("Network input dims without padding {} should be <= padded dims {}".format(
                    (resized_mask_height, resized_mask_width), tuple(mask_t.shape[-2:])))

            # remove extra zeros
            mask_t = mask_t[:, :, :resized_mask_height, :resized_mask_width]

            # resize to original image dims
            if (image_height, image_width) != (mask_t.shape[-2], mask_t.shape[-1]):
                mask_t = F.interpolate(mask_t, (image_height, image_width), mode='bilinear', align_corners=False)

            mask_t = (mask_t > 0.5).long().squeeze(0)

            # replace 1s with instance IDs
            instance_ids_tensor = torch.tensor(active_instances_t, dtype=torch.long, device=device)[:, None, None]
            mask_t = mask_t * instance_ids_tensor

            # assign each pixel to the instance with the longest lifetime
            mask_t = mask_t.max(dim=0)[0]

            for i, instance_id in enumerate(active_instances_t, 1):
                rle_mask = masktools.encode(np.asfortranarray((mask_t == instance_id).cpu().numpy()))["counts"].decode("utf-8")

                instance_rle_masks[instance_id_mapping[instance_id]].append({
                    "frame_id": t,
                    "image_height": image_height,
                    "image_width": image_width,
                    "instance_id": instance_id_mapping[instance_id],
                    "mask": rle_mask
                })

        self.add_sequence_result(sequence, instance_rle_masks, instance_semantic_label_votes)

        return instances_to_keep, {v: k for k, v in instance_id_mapping.items()}

    def add_sequence_result(self, seq, instance_rle_masks, instance_semantic_label_votes):
        # assign semantic label to each instance based on max votes
        for instance_id, instance_attribs in instance_rle_masks.items():
            semantic_label_votes = instance_semantic_label_votes[instance_id]
            max_voted_label, num_votes = max([
                (semantic_label, votes) for semantic_label, votes in semantic_label_votes.items()], key=lambda x: x[1])

            # map the predicted semantic label to the original labels in the Youtube VIS dataset spec
            assert max_voted_label in self.categories, "Label {} does not exist in category ID list".format(max_voted_label)

            for frame_instance in instance_attribs:
                frame_instance["category_id"] = max_voted_label

        os.makedirs(self.results_output_dir, exist_ok=True)
        output_path = os.path.join(self.results_output_dir, "{:04d}.txt".format(int(seq.id)))

        with open(output_path, 'w') as fh:
            for instance_id, instance_attribs in instance_rle_masks.items():

                for frame_instance in instance_attribs:
                    fh.write("{frame_id} {instance_id} {category_id} {img_height} {img_width} {rle_mask}\n".format(
                        frame_id=frame_instance["frame_id"],
                        instance_id=int((frame_instance["category_id"] * 1000) + instance_id),
                        category_id=frame_instance["category_id"],
                        img_height=frame_instance["image_height"],
                        img_width=frame_instance["image_width"],
                        rle_mask=frame_instance["mask"]
                    ))

        if self.save_visualization:
            self.save_sequence_visualizations(seq, instance_rle_masks)

    @Timer.exclude_duration("postprocessing")
    def save_sequence_visualizations(self, seq, instances):
        cmap = create_color_map().tolist()
        seq_output_dir = os.path.join(self.vis_output_dir, "{:04d}".format(int(seq.id)))
        os.makedirs(seq_output_dir, exist_ok=True)

        # arrange the masks according to frame rather than by instance.
        instances_by_frame = defaultdict(list)
        for instance in instances.values():
            for frame_instance in instance:
                instances_by_frame[frame_instance["frame_id"]].append(frame_instance)

        images = seq.load_images()

        for t, image_t in enumerate(images):
            for instance in instances_by_frame[t]:
                category_label = instance["category_id"]
                instance_id = instance["instance_id"]

                color = cmap[instance_id % 256]
                mask = masktools.decode({
                    "size": (instance["image_height"], instance["image_width"]),
                    "counts": instance["mask"]
                })

                annotation_text = "{}{}".format(instance_id, self.category_label[category_label])
                image_t = annotate_instance(image_t, mask, color, annotation_text, font_size=0.25)

            cv2.imwrite(os.path.join(seq_output_dir, '{:05d}.jpg'.format(t)), image_t)

    def save(self, *args, **kwargs):
        print("Applying NMS to results...")
        postprocess_results(results_dir=self.results_output_dir)

    @Timer.exclude_duration("postprocessing")
    def overlay_masks_on_images(self, seq, masks):
        with seq:
            images = seq.images

        assert len(images) == len(masks)
        cmap = create_color_map()

        for t, (image, mask) in enumerate(zip(images, masks)):
            mask = np.array(mask)
            assert mask.shape == image.shape[:2], \
                "Mask has shape {} while image has shape {}".format(mask.shape, image.shape)
            instance_ids = set(np.unique(mask)) - {0}
            assert self.outlier_label not in instance_ids

            for n in instance_ids:
                images[t] = overlay_mask_on_image(images[t], mask == n, mask_color=cmap[n % 256])

        return images
