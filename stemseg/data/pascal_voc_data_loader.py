from torch.utils.data import Dataset

from stemseg.config import cfg
from stemseg.data.generic_image_dataset_parser import parse_generic_image_dataset
from stemseg.data.image_to_seq_augmenter import ImageToSeqAugmenter
from stemseg.structures import BinaryMask, BinaryMaskSequenceList
from stemseg.data.common import compute_resize_params, scale_and_normalize_images
from stemseg.utils import RepoPaths, transforms

import numpy as np
import random
import torch
import torch.nn.functional as F
import os
import yaml


class PascalVOCDataLoader(Dataset):
    def __init__(self, base_dir, ids_json_file, category_agnostic, min_instance_size=50):
        super().__init__()

        self.samples, meta_info = parse_generic_image_dataset(base_dir, ids_json_file)

        def filter_by_mask_area(sample):
            mask_areas = sample.mask_areas()
            instance_idxes_to_keep = [
                i for i in range(len(sample.segmentations)) if mask_areas[i] >= min_instance_size
            ]

            sample.segmentations = [sample.segmentations[i] for i in instance_idxes_to_keep]
            sample.categories = [sample.categories[i] for i in instance_idxes_to_keep]

            return sample

        self.samples = map(filter_by_mask_area, self.samples)

        with open(os.path.join(RepoPaths.dataset_meta_info_dir(), 'pascal_voc.yaml'), 'r') as fh:
            category_details = yaml.load(fh, Loader=yaml.SafeLoader)
            category_details = {cat['id']: cat for cat in category_details}

        if category_agnostic:  # davis
            cat_ids_to_keep = [cat_id for cat_id, attribs in category_details.items() if attribs['keep_davis']]
            self.category_id_mapping = {
                cat_id: 1 for cat_id in cat_ids_to_keep
            }
            self.category_labels = {
                cat_id: 'object' for cat_id in cat_ids_to_keep
            }

        else:  # youtube VIS
            cat_ids_to_keep = [cat_id for cat_id, attribs in category_details.items() if attribs['keep_ytvis']]
            self.category_id_mapping = {
                cat_id: category_details[cat_id]['id_ytvis'] for cat_id in cat_ids_to_keep
            }
            self.category_labels = {
                cat_id: category_details[cat_id]['label_ytvis'] for cat_id in cat_ids_to_keep
            }

        def filter_by_category_id(sample):
            instance_idxes_to_keep = [
                i for i in range(len(sample.segmentations)) if sample.categories[i] in cat_ids_to_keep
            ]

            sample.segmentations = [sample.segmentations[i] for i in instance_idxes_to_keep]
            sample.categories = [sample.categories[i] for i in instance_idxes_to_keep]

            return sample

        self.samples = map(filter_by_category_id, self.samples)

        # remove samples with 0 instances
        self.samples = list(filter(lambda s: len(s.segmentations) > 0, self.samples))

        self.augmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                             rotation_range=(-10, 10), perspective_magnitude=0.08,
                                             hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                             motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                             translate_range=(-0.1, 0.1))

        self.num_frames = cfg.INPUT.NUM_FRAMES
        self.category_agnostic = category_agnostic

        self.np_to_tensor = transforms.BatchImageTransform(transforms.ToTorchTensor(format='CHW'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = sample.load_image()
        image_height, image_width = sample.height, sample.width

        instance_masks = sample.load_masks()
        ignore_mask = sample.load_ignore_mask()

        # apply random horizontal flip
        image, instance_masks, ignore_mask = self.apply_random_flip(image, instance_masks, ignore_mask)

        # convert masks to BinaryMask type
        instance_masks = [BinaryMask(mask) for mask in instance_masks]
        ignore_mask = BinaryMask(ignore_mask)

        # Convert everything to a single element list so that it becomes a one-image 'sequence'
        seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks = \
            [image], [instance_masks], [ignore_mask], [np.zeros((image_height, image_width), np.uint8)]

        # add remaining sequence images by augmenting the original image
        for t in range(self.num_frames - 1):
            # get transformed image, instance mask and point validity mask
            masks = instance_masks + [ignore_mask]
            im_trafo, masks_trafo, invalid_pts = self.augmenter(image, masks)

            instance_masks_trafo, ignore_mask_trafo = masks_trafo[:-1], masks_trafo[-1]

            # add everything to the sequence lists
            seq_images.append(im_trafo)
            seq_instance_masks.append(instance_masks_trafo)
            seq_ignore_masks.append(ignore_mask_trafo)
            seq_invalid_pts_masks.append(invalid_pts)

        # shuffle the elements of the sequence
        seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks = self.apply_random_sequence_shuffle(
            seq_images, seq_instance_masks, seq_ignore_masks, seq_invalid_pts_masks)

        # convert images to tensors
        seq_images = torch.stack(self.np_to_tensor(*seq_images), 0).float()

        # scale and normalize images
        seq_images = scale_and_normalize_images(seq_images, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD,
                                                not cfg.INPUT.BGR_INPUT, cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)

        seq_invalid_pts_masks = [torch.from_numpy(mask).float() for mask in seq_invalid_pts_masks]

        for i in range(len(seq_images)):
            invalid_pts = 1. - seq_invalid_pts_masks[i][None, :, :]
            seq_images[i] = seq_images[i] * invalid_pts

        # resize images to the required input size
        new_width, new_height, scale_factor = compute_resize_params(image, cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
        seq_images = F.interpolate(seq_images, (new_height, new_width), mode='bilinear', align_corners=False)

        # resize masks to the required input size
        seq_instance_masks = BinaryMaskSequenceList(seq_instance_masks)
        seq_instance_masks = seq_instance_masks.resize((new_width, new_height))
        seq_ignore_masks = [mask.resize((new_height, new_width)) for mask in seq_ignore_masks]

        # convert masks to torch tensors
        seq_instance_masks = seq_instance_masks.tensor().permute(1, 0, 2, 3)  # [N, T, H, W]
        seq_ignore_masks = torch.stack([mask.tensor() for mask in seq_ignore_masks], 0)  # [T, H, W]

        instance_category_ids = torch.tensor([self.category_id_mapping[cat_id] for cat_id in sample.categories])
        category_labels = [self.category_labels[cat_id] for cat_id in sample.categories]

        # combine everything into a dictionary
        targets = {"masks": seq_instance_masks,
                   "category_ids": instance_category_ids,
                   "labels": instance_category_ids,
                   'ignore_masks': seq_ignore_masks}

        return seq_images, targets, (image_width, image_height), {"category_labels": category_labels}

    def apply_random_flip(self, image, instance_masks, ignore_mask):
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            instance_masks = [np.flip(instance_mask, axis=1) for instance_mask in instance_masks]
            ignore_mask = np.flip(ignore_mask, axis=1)

        return image, instance_masks, ignore_mask

    def apply_random_sequence_shuffle(self, images, instance_masks, ignore_masks, invalid_pt_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)

        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        ignore_masks = [ignore_masks[i] for i in perm]
        invalid_pt_masks = [invalid_pt_masks[i] for i in perm]

        return images, instance_masks, ignore_masks, invalid_pt_masks
