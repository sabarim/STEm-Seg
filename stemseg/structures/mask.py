import cv2
import numpy as np
import torch
import torch.nn.functional as F


class BinaryMask(object):
    def __init__(self, mask):
        if isinstance(mask, np.ndarray):
            self._mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.uint8)
        else:
            assert isinstance(mask, torch.Tensor)
            self._mask = mask.to(torch.uint8)

        assert self._mask.ndimension() == 2, "Provided mask has {} dimensions instead of 2".format(self._mask.ndimension())

    def __getitem__(self, index):
        return self._mask[index]

    def copy(self):
        return self.__class__(self._mask.clone())

    def to(self, *args, **kwargs):
        return self.__class__(self._mask.to(*args, **kwargs))

    def cuda(self):
        return self.to("cuda:0")

    def cpu(self):
        return self.to("cpu")

    def resize(self, size=None, scale_factor=None):
        # Size should be (height, width) here
        assert not (size is None and scale_factor is None)
        assert not (size is not None and scale_factor is not None)

        # resized_mask = F.interpolate(self._mask[None, None].float(), size, scale_factor, mode='nearest').byte()[0, 0]
        resized_mask = F.interpolate(self._mask[None, None].float(), size, scale_factor, mode='bilinear', align_corners=False)
        resized_mask = (resized_mask > 0.5).byte()[0, 0]
        return self.__class__(resized_mask)

    def pad(self, new_width, new_height):
        mask_expanded = self._mask[None, None, :, :]
        pad_right, pad_bottom = new_width - self.width, new_height - self.height
        padded = F.pad(mask_expanded, (0, pad_right, 0, pad_bottom)).squeeze(0).squeeze(0)

        return self.__class__(padded)

    def crop(self, xmin, ymin, xmax, ymax):
        assert 0 <= ymin < ymax < self._mask.shape[0], "Invalid y-coords for crop ({}, {}) for mask of size {}".format(
            ymin, ymax, str(self._mask.shape))
        assert 0 <= xmin < xmax < self._mask.shape[1], "Invalid x-coords for crop ({}, {}) for mask of size {}".format(
            xmin, xmax, str(self._mask.shape))

        return self.__class__(self._mask[ymin:ymax, xmin:xmax])

    def flip_horizontal(self):
        return self.__class__(self._mask.flip(dims=[1]))

    def transform_affine(self, transformation_matrix):
        """
        :param transformation_matrix: should be a (2x3) NumPy array
        :return:
        """
        mask_np = self._mask.detach().cpu().numpy()
        mask_np = cv2.warpAffine(mask_np, transformation_matrix, (self.width, self.height),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return self.__class__(mask_np)

    def translate(self, tx, ty):
        padding = (tx, -tx, ty, -ty)
        mask = F.pad(self._mask[None, None, ...], padding).squeeze(0).squeeze(0)
        return self.__class__(mask)

    def tensor(self, copy=False):
        if copy:
            return self._mask.clone()
        else:
            return self._mask

    def bounding_box(self, return_none_if_invalid):
        reduced_y = torch.any(self._mask, dim=0)
        reduced_x = torch.any(self._mask, dim=1)

        xmax = reduced_y.argmax()
        if reduced_y.sum() == 0:  # mask is all zeros
            if return_none_if_invalid:
                return None
            else:
                return Box([-1, -1, 0, 0], 'xyxy', False)

        xmin = reduced_y.numel() - torch.flip(reduced_y, dims=[0]).argmax()
        ymax = reduced_x.argmax()
        ymin = reduced_x.numel() - torch.flip(reduced_x, dims=[0]).argmax()

        return Box((xmin, ymin, xmax, ymax), 'xyxy', True)

    height = property(fget=lambda self: self._mask.shape[0])
    width = property(fget=lambda self: self._mask.shape[1])
    shape = property(fget=lambda self: (self.height, self.width))


class BinaryMaskSequenceList(object):
    def __init__(self, mask_sequence_list):
        self._mask_sequence_list = mask_sequence_list

        len0 = len(self._mask_sequence_list[0])
        for t in range(1, len(self._mask_sequence_list)):
            assert len(self._mask_sequence_list[t]) == len0

    def to(self, *args, **kwargs):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.to(*args, **kwargs) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def cuda(self):
        return self.to("cuda:0")

    def cpu(self):
        return self.to("cpu")

    def copy(self):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.copy() for mask in instance_list])
        return self.__class__(mask_sequence_list)

    def resize(self, size=None, scale_factor=None):
        """
        Resizes all masks in the sequence list
        :param size: New dimensions in (W, H) format
        :param scale_factor: Alternatively, a scale factor for resizing
        :return:
        """
        if size is not None:
            size = size[::-1]

        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.resize(size, scale_factor) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def pad(self, new_width, new_height):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.pad(new_width, new_height) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def crop(self, xmin, ymin, xmax, ymax):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.crop(xmin, ymin, xmax, ymax) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def flip_horizontal(self):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.flip_horizontal() for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def transform_affine(self, transformation_matrix):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.transform_affine(transformation_matrix) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def translate(self, tx, ty, frames=None):
        mask_sequence_list = []
        for instance_list in self._mask_sequence_list:
            mask_sequence_list.append([mask.translate(tx, ty) for mask in instance_list])

        return self.__class__(mask_sequence_list)

    def reverse(self):
        return self.__class__(self._mask_sequence_list[::-1])

    def reorder(self, perm):
        assert len(perm) == self.num_frames
        return self.__class__([self._mask_sequence_list[i] for i in perm])

    def bounding_boxes(self):
        box_sequence_list = []
        for instance_list in self._mask_sequence_list:
            box_sequence_list.append([mask.bounding_box(False) for mask in instance_list])

        return BoxSequenceList(box_sequence_list)

    def tensor(self, format='TN'):
        assert format in ['TN', 'NT']

        tensor_list = []
        for instance_list in self._mask_sequence_list:
            tensor_list.append(torch.stack([mask.tensor() for mask in instance_list]))

        if format == 'TN':
            return torch.stack(tensor_list)
        else:
            return torch.stack(tensor_list).permute(1, 0, 2, 3)

    num_frames = property(fget=lambda self: len(self._mask_sequence_list))
    num_instances = property(fget=lambda self: len(self._mask_sequence_list[0]))
    shape = property(fget=lambda self: (self.num_frames, self.num_instances) + self._mask_sequence_list[0][0].shape)
