from stemseg.config import cfg
from stemseg.data.video_dataset import VideoDataset
from stemseg.structures.mask import BinaryMask, BinaryMaskSequenceList

import math
import numpy as np
import random


class MOTSDataLoader(VideoDataset):
    IGNORE_MASK_CAT_ID = 3

    def __init__(self, base_dir, vds_json_file, samples_to_create,
                 apply_augmentation=False):
        super(MOTSDataLoader, self).__init__(base_dir, vds_json_file, cfg.INPUT.NUM_FRAMES, apply_augmentation)

        # filtering zero instance frames introduces very long frame gaps for some videos. It is therefore better to
        # break up such cases into multiple sequences so that a single training sample does not contain large temporal
        # gaps.
        split_sequences = []

        for seq in self.sequences:
            suffix = 1  # for keeping the sequence ID unique
            current_gap_len = 0
            current_seq_frame_idxes = []

            for t in range(len(seq)):
                instance_cats_t = set([seq.instance_categories[iid] for iid in seq.segmentations[t].keys()])

                if len(instance_cats_t - {self.IGNORE_MASK_CAT_ID}) == 0:  # no car or pedestrian instances
                    current_gap_len += 1
                    if current_gap_len == 6 and current_seq_frame_idxes:
                        split_sequences.append(seq.extract_subsequence(current_seq_frame_idxes,
                                                                       "{}_{}".format(seq.id, str(suffix))))
                        suffix += 1
                        current_seq_frame_idxes = []
                else:
                    current_gap_len = 0
                    current_seq_frame_idxes.append(t)

            if current_seq_frame_idxes:
                split_sequences.append(seq.extract_subsequence(current_seq_frame_idxes,
                                                               "{}_{}".format(seq.id, str(suffix))))

        self.sequences = split_sequences

        assert samples_to_create > 0, "Number of training samples is required for train mode"
        self.samples = self.create_training_subsequences(samples_to_create)

    def create_training_subsequences(self, num_subsequences):
        frame_range = list(range(cfg.DATA.KITTI_MOTS.FRAME_GAP_LOWER, cfg.DATA.KITTI_MOTS.FRAME_GAP_UPPER + 1))
        subseq_length = self.clip_length

        # filter sequences which are too short
        min_sequence_length = frame_range[0] + 1  # so that multiple, different subsequences can be generated
        sequences = [seq for seq in self.sequences if len(seq) > min_sequence_length]
        # print("Num sequences: {} -> {}".format(len(dataset.sequences), len(sequences)))

        # compute number of sub-sequences to create from each video sequence
        total_frames = sum([len(seq) for seq in sequences])
        samples_per_seq = [max(1, int(math.ceil((len(seq) / total_frames) * num_subsequences))) for seq in sequences]

        subseq_span_range = frame_range.copy()
        subsequence_idxes = []

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
        subsequences = [sequences[video_id].extract_subsequence(frame_idxes) for video_id, frame_idxes in subsequence_idxes]

        return subsequences

    def parse_sample_at(self, idx):
        sample = self.samples[idx]

        images = sample.load_images()
        masks = sample.load_masks()  # list(T))

        instance_categories = sample.category_labels

        # separate the ignore masks
        if 3 in instance_categories:
            ignore_mask_idx = instance_categories.index(self.IGNORE_MASK_CAT_ID)
            instance_categories.remove(self.IGNORE_MASK_CAT_ID)

            ignore_masks = [BinaryMask(masks_t[ignore_mask_idx]) for masks_t in masks]

            other_idxes = list(range(len(sample.instance_ids)))
            other_idxes.remove(ignore_mask_idx)
            masks = [
                [
                    BinaryMask(masks_t[i])
                    for i in other_idxes
                ]
                for masks_t in masks
            ]

        else:
            height, width = images[0].shape[:2]
            ignore_masks = [BinaryMask(np.zeros((height, width), np.uint8)) for _ in range(len(images))]

            masks = [
                [
                    BinaryMask(mask) for mask in masks_t
                ]
                for masks_t in masks
            ]

        masks = BinaryMaskSequenceList(masks)

        if masks.num_instances == 0:
            raise ValueError("No instances exist in the masks (seq: {})".format(sample.id))

        return images, masks, instance_categories, {'seq_name': sample.id, 'ignore_masks': ignore_masks}

    def __len__(self):
        return len(self.samples)
