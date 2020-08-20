from stemseg.config import cfg
from stemseg.utils import transforms

import math
import torch


class ImageList(object):
    def __init__(self, images_tensor, image_sizes, original_image_sizes=None):
        self.tensors = images_tensor  # [N, T, C, H, W]
        self.image_sizes = image_sizes  # (H, W)
        self.original_image_sizes = original_image_sizes  # (W, H)

        norm_factor = 255.0 if cfg.INPUT.NORMALIZE_TO_UNIT_SCALE else 1.0
        self.__reverse_preprocessing = transforms.Compose(
                [transforms.ReverseNormalize(norm_factor, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD),
                 transforms.Identity() if cfg.INPUT.BGR_INPUT else transforms.ReverseColorChannels()]
            )

    def to(self, *args, **kwargs):
        cast_tensors = self.tensors.to(*args, **kwargs)
        return self.__class__(cast_tensors, self.image_sizes, self.original_image_sizes)

    def cuda(self):
        cuda_tensors = self.tensors.cuda()
        return self.__class__(cuda_tensors, self.image_sizes, self.original_image_sizes)

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, index):
        assert index <= self.num_seqs
        return self.tensors[index]

    def numpy(self, seq_idxs=None, t_idxs=None):
        """
        Returns the images as NumPy tensors a list of lists (first index: seq, second index: time)
        :param seq_idxs:
        :param t_idxs:
        :return:
        """
        if seq_idxs is None:
            seq_idxs = list(range(self.num_seqs))
        if t_idxs is None:
            t_idxs = list(range(self.num_frames))

        sequences = []
        for i in seq_idxs:
            seq_images = []
            for t in t_idxs:
                image = self.tensors[i, t].permute(1, 2, 0).detach().cpu().numpy()
                image = self.__reverse_preprocessing(image)
                seq_images.append(image)
            sequences.append(seq_images)

        return sequences

    @classmethod
    def from_image_sequence_list(cls, image_sequence_list, original_dims=None):
        """
        Converts a list of image sequences to an ImageList object
        :param image_sequence_list: Assuming there are N sequences, each of length T, this argument should be a list
        containing N sub-lists, each of length T.
        :param original_dims: The original sizes (WH) of the images before (rescaling/padding according to model input
        requirements).
        """
        source_dtype = image_sequence_list[0][0].dtype
        assert source_dtype in (torch.uint8, torch.float32, torch.float64), \
            "Array type should either be float32 or float64, encountered: {}".format(source_dtype)

        max_height = max_width = -1
        image_sizes = []
        seq_length = -1

        for image_sequence in image_sequence_list:
            seq_length = len(image_sequence) if seq_length == -1 else seq_length
            if len(image_sequence) != seq_length:
                raise ValueError("All sequences must contain the same number of images. Found {} and {}".format(
                    seq_length, len(image_sequence)))

            seq_heights = [im.shape[1] for im in image_sequence]
            seq_widths = [im.shape[2] for im in image_sequence]
            assert len(set(seq_widths)) == 1 and len(set(seq_heights)) == 1, \
                "All images within a given sequence must have the same size. Encountered sizes: {}".format(
                    ", ".join(["(%d, %d)" % (im.shape[1], im.shape[0]) for im in image_sequence]))

            height, width = seq_heights[0], seq_widths[0]
            image_sizes.append((height, width))

            max_height = max(height, max_height)
            max_width = max(width, max_width)

        # make width and height a multiple of 32
        max_width = (int(math.ceil(max_width / 32)) * 32) + 0
        max_height = (int(math.ceil(max_height / 32)) * 32) + 0

        num_sequences = len(image_sequence_list)
        tensors = torch.zeros(len(image_sequence_list), seq_length, 3, max_height, max_width, dtype=source_dtype)

        for i in range(num_sequences):
            height, width = image_sizes[i]

            for j in range(seq_length):
                tensors[i, j, :, :height, :width] = image_sequence_list[i][j]

        tensors = tensors.detach()
        return cls(tensors, image_sizes, original_dims)

    num_frames = property(fget=lambda self: self.tensors.shape[1])
    num_seqs = property(fget=lambda self: self.tensors.shape[0])
    max_size = property(fget=lambda self: self.tensors.shape[-2:])  # (H, W)
