from PIL import Image
from stemseg.config import cfg
from stemseg.data.common import compute_resize_params_2
from stemseg.utils.timer import Timer
from stemseg.utils.vis import create_color_map, overlay_mask_on_image

import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F


def compute_pixel_box_area_ratio(mask):
    y_coords, x_coords = mask.nonzero().unbind(1)
    if y_coords.numel() == 0:
        return 0.0

    pixel_area = mask.sum(dtype=torch.float32).item()

    box_area = (y_coords.max() - y_coords.min()) * (x_coords.max() - x_coords.min())
    if box_area == 0:
        return 0.

    return pixel_area / box_area.item()


class DavisOutputGenerator(object):
    def __init__(self, output_dir, outlier_label, save_visualization, *args, **kwargs):
        self.results_output_dir = os.path.join(output_dir, "results")
        self.vis_output_dir = os.path.join(output_dir, "vis")

        self.outlier_label = outlier_label
        self.save_visualization = save_visualization
        self.upscaled_inputs = kwargs.get("upscaled_inputs")

    @Timer.exclude_duration("postprocessing")
    def process_sequence(self, sequence, track_mask_idxes, track_mask_labels, instance_pt_counts, instance_lifetimes,
                         category_masks, mask_dims, mask_scale, max_tracks, device="cpu"):
        """
        Given a list of mask indices per frame, creates a sequence of masks for the entire sequence.
        :param track_mask_idxes: list(tuple(tensor, tensor))
        :param track_mask_labels: list(tensor)
        :param instance_pt_counts: dict(int -> int)
        :param category_masks: irrelevant
        :param mask_dims: tuple(int, int) (height, width)
        :param mask_scale: int
        :param max_tracks: int
        :param device: str
        :return: list(PIL.Image)
        """
        mask_height, mask_width = mask_dims
        image_height, image_width = sequence.image_dims

        assert len(track_mask_idxes) == len(track_mask_labels)
        assert max_tracks < 256

        instances_to_keep = [
                                instance_id for instance_id, _ in sorted(
                [(k, v) for k, v in instance_lifetimes.items()], key=lambda x: x[1], reverse=True
            ) if instance_id != self.outlier_label
        ]

        instances_to_keep = instances_to_keep[:max_tracks]
        num_tracks = len(instances_to_keep)

        print("Number of instances: ", len(instances_to_keep))

        # move tensors to the target device
        track_mask_labels = [x.to(device=device) for x in track_mask_labels]
        track_mask_idxes = [(coords[0].to(device=device), coords[1].to(device=device)) for coords in track_mask_idxes]

        masks = []
        cmap = create_color_map().flatten()

        for t in range(len(track_mask_idxes)):
            mask_t = torch.zeros(mask_height, mask_width, dtype=torch.long, device=device)
            mask_t[track_mask_idxes[t]] = track_mask_labels[t]

            mask_t = torch.stack([mask_t == ii for ii in instances_to_keep], 0)

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

            mask_condensed = torch.zeros(image_height, image_width, dtype=torch.uint8, device=device)
            for n in range(num_tracks):
                mask_condensed = torch.where(mask_t[n], torch.tensor(n + 1, dtype=torch.uint8, device=device),
                                             mask_condensed)

            mask_condensed = Image.fromarray(mask_condensed.cpu().numpy())
            mask_condensed.putpalette(cmap)
            masks.append(mask_condensed)

        seq_results_dir = os.path.join(self.results_output_dir, sequence.id)
        os.makedirs(seq_results_dir, exist_ok=True)

        for t, mask in enumerate(masks):
            mask.save(os.path.join(seq_results_dir, "{:05d}.png".format(t)))

        if not self.save_visualization:
            return instances_to_keep, dict()

        seq_vis_dir = os.path.join(self.vis_output_dir, sequence.id)
        os.makedirs(seq_vis_dir, exist_ok=True)

        overlayed_images = self.overlay_masks_on_images(sequence, masks)
        for t, overlayed_image in enumerate(overlayed_images):
            cv2.imwrite(os.path.join(seq_vis_dir, "{:05d}.jpg".format(t)), overlayed_image)

        return instances_to_keep, dict()

    def save(self, *args, **kwargs):
        pass

    @Timer.exclude_duration("postprocessing")
    def overlay_masks_on_images(self, seq, masks):
        # with seq:
        #     images = seq.images
        images = seq.load_images()

        assert len(images) == len(masks), "Got {} images but {} masks".format(len(images), len(masks))
        cmap = create_color_map()

        for t, (image, mask) in enumerate(zip(images, masks)):
            mask = np.array(mask)

            assert mask.shape == image.shape[:2], \
                "Mask has shape {} while image has shape {}".format(mask.shape, image.shape)
            instance_ids = set(np.unique(mask)) - {0}
            assert self.outlier_label not in instance_ids

            for n in instance_ids:
                images[t] = overlay_mask_on_image(images[t], mask == n, mask_color=cmap[n])

        return images
