from pycocotools import mask as masktools

import cv2
import json
import os
import numpy as np


def parse_generic_image_dataset(base_dir, dataset_json):
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}
    samples = [GenericImageSample(base_dir, sample) for sample in dataset["images"]]

    return samples, meta_info


class GenericImageSample(object):
    def __init__(self, base_dir, sample):
        self.height = sample['height']
        self.width = sample['width']
        self.path = os.path.join(base_dir, sample['image_path'])
        self.categories = [int(cat_id) for cat_id in sample['categories']]
        self.segmentations = sample['segmentations']
        self.ignore = sample.get("ignore", None)

    def mask_areas(self):
        rle_objs = [{
            "size": (self.height, self.width),
            "counts": seg.encode("utf-8")
        } for seg in self.segmentations ]

        return [masktools.area(obj) for obj in rle_objs]

    def load_image(self):
        im = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("No image found at path: {}".format(self.path))
        return im

    def load_ignore_mask(self):
        if self.ignore is None:
            return None

        return np.ascontiguousarray(masktools.decode({
            "size": (self.height, self.width),
            "counts": self.ignore.encode('utf-8')
        }).astype(np.uint8))

    def load_masks(self):
        return [np.ascontiguousarray(masktools.decode({
            "size": (self.height, self.width),
            "counts": seg.encode('utf-8')
        }).astype(np.uint8)) for seg in self.segmentations]

    def filter_categories(self, cat_ids_to_keep):
        self.categories, self.segmentations = zip(*[
            (cat_id, seg) for cat_id, seg in zip(self.categories, self.segmentations) if cat_id in cat_ids_to_keep
        ])
