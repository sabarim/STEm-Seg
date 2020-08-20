from stemseg.utils.vis import overlay_mask_on_image

import cv2
import numpy as np


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


def annotate_instance(image, mask, color, text_label, font_size=0.5, draw_bbox=True):
    """
    :param image: np.ndarray(H, W, 3)
    :param mask: np.ndarray(H, W)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param text_label: str
    :param font_size
    :param draw_bbox: bool
    :return: np.ndarray(H, W, 3)
    """
    assert image.shape[:2] == mask.shape, "Shape mismatch between image {} and mask {}".format(image.shape, mask.shape)
    color = tuple(color)

    overlayed_image = overlay_mask_on_image(image, mask, mask_color=color)
    bbox = bbox_from_mask(mask)
    if not bbox:
        return overlayed_image

    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(overlayed_image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)

    (text_width, text_height), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness=1)
    text_offset_x, text_offset_y = int(xmin + 2), int(ymin + text_height + 2)

    text_bg_box_pt1 = int(text_offset_x), int(text_offset_y + 2)
    # print(text_offset_x, text_width, text_offset_y, text_height)
    text_bg_box_pt2 = int(text_offset_x + text_width + 2), int(text_offset_y - text_height - 2)

    if draw_bbox:
        cv2.rectangle(overlayed_image, text_bg_box_pt1, text_bg_box_pt2, color=(255, 255, 255), thickness=-1)

    cv2.putText(overlayed_image, text_label, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0))

    return overlayed_image
