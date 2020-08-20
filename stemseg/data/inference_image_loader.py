from stemseg.config import cfg
from stemseg.structures import ImageList
from stemseg.utils import transforms
from torch.utils.data import Dataset
from stemseg.data.common import scale_and_normalize_images, compute_resize_params_2

import cv2
import numpy as np
import torch.nn.functional as F


class InferenceImageLoader(Dataset):
    def __init__(self, images):
        super().__init__()

        self.np_to_tensor = transforms.ToTorchTensor(format='CHW')

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Unexpected type for image: {}".format(type(image)))

        image_height, image_width = image.shape[:2]

        # convert image to tensor
        image = self.np_to_tensor(image).float()

        # resize image
        new_width, new_height, _ = compute_resize_params_2((image_width, image_height), cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
        image = F.interpolate(image.unsqueeze(0), (new_height, new_width), mode='bilinear', align_corners=False)

        # compute scale factor for image resizing
        image = scale_and_normalize_images(image, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD, not cfg.INPUT.BGR_INPUT, cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)

        return image.squeeze(0), (image_width, image_height), index


def collate_fn(samples):
    image_seqs, original_dims, idxes = zip(*samples)
    image_seqs = [[im] for im in image_seqs]
    image_seqs = ImageList.from_image_sequence_list(image_seqs, original_dims)
    return image_seqs, idxes
