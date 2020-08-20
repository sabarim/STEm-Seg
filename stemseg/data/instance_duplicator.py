import cv2
import numpy as np
import random


class InstanceDuplicator(object):
    def __init__(self):
        pass

    def __call__(self, images, masks):
        """
        :param images: List of T images as numpy arrays in [H, W, 3] (BGR) format
        :param masks: list of T instance masks as numpy arrays of shape [H, W] and dtype uint8
        :return: list of T images and masks with duplicated instance
        """
        # lots of potentially unstable stuff happening here.
        try:
            return self._augment(images, masks)
        except Exception as err:
            print("Exception occurred trying to duplicate instance")
            print(err)
            return None, None

    @staticmethod
    def bbox_from_mask(mask):
        reduced_y = np.any(mask, axis=0)
        reduced_x = np.any(mask, axis=1)

        x_min = reduced_y.argmax()
        if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
            return None

        x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

        y_min = reduced_x.argmax()
        y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

        return x_min, y_min, x_max, y_max

    @staticmethod
    def _augment(images, masks):
        modified_original_instance_masks = []
        duplicate_instance_masks = []
        duplicated_instance_images = []

        boxes = []
        bbox_widths = []
        horiz_multiplier = None
        vert_multiplier = None

        touches_left_boundary, touches_right_boundary = False, False
        touches_top_boundary, touches_bottom_boundary = False, False

        for mask in masks:
            mask_height, mask_width = mask.shape

            # extract bounding box from mask
            bbox = InstanceDuplicator.bbox_from_mask(mask)
            boxes.append(bbox)

            if bbox is None:
                continue

            xmin, ymin, xmax, ymax = bbox
            bbox_widths.append(xmax - xmin)

            # check if the mask touches the left/right boundaries
            if xmin == 0:
                touches_left_boundary = True
            if xmax == mask_width:
                touches_right_boundary = True

            # check if the mask touches the top/bottom boundaries
            if ymin == 0:
                touches_top_boundary = True
            if ymax == mask_height:
                touches_bottom_boundary = True

            # if the instance has large width...
            if xmax - xmin > 0.4 * mask_width:
                if xmin == 0:
                    # ...and it touches the left boundary, only allow it to be shifted further to the left
                    horiz_multiplier = -1.
                elif xmax == mask_width:
                    # ...and it touches the right boundary, only allow it to be shifted further to the right
                    horiz_multiplier = 1.

            # if the instance has small width...
            elif xmax - xmin < 0.2 * mask_width:
                xc = (xmin + xmax) / 2.

                # ...and it is close to the left boundary
                if xc < mask_width * 0.25:
                    # it might disappear if moved to the left. Hence, only allow it to be moved to the right
                    horiz_multiplier = 1.

                # ...and it is close to the right boundary
                elif xc > mask_width * 0.75:
                    # it might disappear if moved to the right. Hence, only allow it to be moved to the left
                    horiz_multiplier = -1.

            # --- REPEAT THE ABOVE LOGIC FOR THE VERTICAL AXIS ---

            # if the instance has large height...
            if ymax - ymin > 0.4 * mask_height:
                if ymin == 0:
                    # ...and it touches the top boundary, only allow it to be shifted further to the top
                    vert_multiplier = -1.
                elif ymax == mask_height:
                    # ...and it touches the bottom boundary, only allow it to be shifted further to the bottom
                    vert_multiplier = 1.

            # if the instance has small height...
            elif ymax - ymin < 0.2 * mask_height:
                yc = (ymin + ymax) / 2.

                # ...and it is close to the top boundary
                if yc < mask_height * 0.25:
                    # it might disappear if moved to the top. Hence, only move to the bottom
                    vert_multiplier = 1.

                # ...and it is close to the bottom boundary
                elif yc > mask_height * 0.75:
                    # it might disappear if moved to the bottom. Hence, only move to the top
                    vert_multiplier = -1.

        # if the mask touches both the left and right boundaries, duplication is not feasible
        if touches_left_boundary and touches_right_boundary:
            return None, None

        # if the mask touches either the left or right boundary, flipping the duplicated instance is infeasible
        flipping_feasible = (not touches_left_boundary) and (not touches_right_boundary)

        # if the mask touches both the top and bottom boundaries, vertical shift should not be applied
        if touches_bottom_boundary and touches_top_boundary:
            vert_multiplier = 0.

        if horiz_multiplier is None:
            horiz_multiplier = -1 if random.random() < 0.5 else 1.

        if vert_multiplier is None:
            vert_multiplier = -1 if random.random() < 0.5 else 1.

        flip = random.random() < 0.5 if flipping_feasible else False

        for image, mask, bbox in zip(images, masks, boxes):
            assert image.shape[:2] == mask.shape
            img_height, img_width = image.shape[:2]

            if bbox is None:
                duplicate_instance_masks.append(np.copy(mask))
                modified_original_instance_masks.append(mask)
                duplicated_instance_images.append(np.copy(image))
                continue

            xmin, ymin, xmax, ymax = bbox
            width, height = xmax - xmin, ymax - ymin

            if flip:
                shifted_image = np.copy(image)
                shifted_mask = np.copy(mask)
                shifted_image[ymin:ymax, xmin:xmax] = np.flip(shifted_image[ymin:ymax, xmin:xmax], axis=1)
                shifted_mask[ymin:ymax, xmin:xmax] = np.flip(shifted_mask[ymin:ymax, xmin:xmax], axis=1)
            else:
                shifted_image = image
                shifted_mask = mask

            shift_x = horiz_multiplier * ((width * 0.75) + (random.random() * 0.25 * width))
            shift_y = vert_multiplier * (height * random.random() * 0.25)

            shift_x = min(shift_x, img_width * 0.3)
            shift_y = min(shift_y, img_height * 0.3)

            affine_mat = np.array([[1., 0., shift_x],
                                   [0., 1., shift_y]], np.float32)

            shifted_image = cv2.warpAffine(shifted_image, affine_mat, (img_width, img_height))
            shifted_mask = cv2.warpAffine(shifted_mask, affine_mat, (img_width, img_height))

            shifted_mask = np.stack([shifted_mask] * 3, axis=2)  # [H, W, 3]

            # copy duplicated instance
            duplicated_image = np.where(shifted_mask > 0, shifted_image, image)

            duplicated_instance_images.append(duplicated_image)

            shifted_mask = shifted_mask[:, :, 0]
            duplicate_instance_masks.append(shifted_mask)

            modified_original_mask = np.where(shifted_mask, 0, mask)
            modified_original_instance_masks.append(modified_original_mask)

        return duplicated_instance_images, [modified_original_instance_masks, duplicate_instance_masks]
