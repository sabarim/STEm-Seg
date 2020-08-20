from stemseg.config import cfg
from stemseg.data.video_dataset import VideoDataset
from stemseg.data.instance_duplicator import InstanceDuplicator
from stemseg.structures.mask import BinaryMask, BinaryMaskSequenceList

import math
import numpy as np
import random


class DavisDataLoader(VideoDataset):
    def __init__(self, base_dir, vds_json_file,
                 samples_to_create=-1,
                 apply_augmentation=False,
                 single_instance_duplication=False,
                 background_as_ignore_region=True):

        super(DavisDataLoader, self).__init__(base_dir, vds_json_file, cfg.INPUT.NUM_FRAMES, apply_augmentation)

        self.filter_zero_instance_frames()

        self.samples = self.create_training_subsequences(samples_to_create)

        self.instance_duplicator = InstanceDuplicator()
        self.single_instance_duplication = single_instance_duplication
        self.background_as_ignore_region = background_as_ignore_region

    def create_training_subsequences(self, num_subsequences):
        frame_range = list(range(cfg.DATA.DAVIS.FRAME_GAP_LOWER, cfg.DATA.DAVIS.FRAME_GAP_UPPER + 1))
        subseq_length = self.clip_length

        # filter sequences which are too short
        min_sequence_length = frame_range[0] + 1  # so that multiple, different subsequences can be generated
        sequences = [seq for seq in self.sequences if len(seq) > min_sequence_length]

        # compute number of sub-sequences to create from each video sequence
        total_frames = sum([len(seq) for seq in sequences])
        samples_per_seq = [max(1, int(math.ceil((len(seq) / total_frames) * num_subsequences))) for seq in sequences]

        subseq_span_range = frame_range.copy()
        subsequence_idxes = []
        # num_intermediate_frames = subseq_length - 2

        for sequence, num_samples in zip(sequences, samples_per_seq):
            for _ in range(num_samples):
                subseq_span = min(random.choice(subseq_span_range), len(sequence) - 1)
                max_start_idx = len(sequence) - subseq_span - 1
                assert max_start_idx >= 0

                start_idx = 0 if max_start_idx == 0 else random.randint(0, max_start_idx)
                end_idx = start_idx + subseq_span
                sample_idxes = np.round(np.linspace(start_idx, end_idx, subseq_length)).astype(np.int32).tolist()

                assert len(set(sample_idxes)) == len(sample_idxes)  # sanity check: ascertain no duplicate indices
                subsequence_idxes.append((sequence.id, sample_idxes))

        # because of rounding up the number of samples to create per sequence, we will always have more than the
        # required number of samples. So we randomly select the required number.
        assert len(subsequence_idxes) >= num_subsequences, \
            "{} should be >= {}".format(len(subsequence_idxes), num_subsequences)

        subsequence_idxes = random.sample(subsequence_idxes, num_subsequences)
        random.shuffle(subsequence_idxes)

        sequences = {seq.id: seq for seq in sequences}
        subsequences = [
            sequences[video_id].extract_subsequence(frame_idxes)
            for video_id, frame_idxes in subsequence_idxes
        ]

        return subsequences

    def parse_sample_at(self, idx):
        sample = self.samples[idx]

        images = sample.load_images()
        masks = sample.load_masks()  # list(T))

        if len(sample.instance_ids) == 1 and self.single_instance_duplication:
            masks_flat = [mask[0] for mask in masks]
            augmented_images, augmented_masks = self.instance_duplicator(images, masks_flat)
            if augmented_images is not None:  # duplication was successful
                images = augmented_images
                masks = list(zip(*augmented_masks))  # list(N, list(T)) --> list(T, list(N))

        if self.background_as_ignore_region:
            fg_masks = [np.any(np.stack(masks_t, 0), 0) for masks_t in masks]
            ignore_masks = [BinaryMask((fg_mask == 0).astype(np.uint8)) for fg_mask in fg_masks]
        else:
            ignore_masks = [BinaryMask(np.zeros_like(masks[0][0], np.uint8)) for _ in range(len(masks))]

        masks = [
            [BinaryMask(mask) for mask in masks_t]
            for masks_t in masks
        ]

        masks = BinaryMaskSequenceList(masks)
        instance_categories = [1 for _ in range(masks.num_instances)]

        return images, masks, instance_categories, {"ignore_masks": ignore_masks, "seq_name": sample.id}

    def __len__(self):
        return len(self.samples)
