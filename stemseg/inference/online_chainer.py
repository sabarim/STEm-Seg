from collections import defaultdict

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F


def masks_to_coord_list(masks):
    """
    :param masks: tensor(T, H, W)
    :return: list(tuple(tensor(M), tensor(M)))
    """
    fg_idxes_all = []
    for t in range(masks.shape[0]):
        fg_idxes = masks[t].nonzero()  # [M, 2]
        fg_idxes = tuple(fg_idxes.unbind(1))
        fg_idxes_all.append(fg_idxes)

    return fg_idxes_all


class TrackContainer(object):
    """
    Container for holding the final stitched labels assigned to every instance in a video sequence.
    """
    def __init__(self, num_frames):
        self._frame_labels = [None for _ in range(num_frames)]
        self._is_frozen = [False for _ in range(num_frames)]

        self._highest_instance_id = 0

    def add_labels(self, frame_nums, labels):
        """
        Assign labels to the foreground pixels of a given frame
        :param frame_nums: list(int)
        :param labels: list(tensor(N, E)). These should be the global track labels and not the cluster labels within a
        given sub-sequence.
        :return: The next available instance ID.
        """
        assert all([self._frame_labels[t] is None for t in frame_nums])
        for t, labels_t in zip(frame_nums, labels):
            self._frame_labels[t] = labels_t
            if labels_t.numel() > 0:
                self._highest_instance_id = max(self._highest_instance_id, labels_t.max().item())

        return self._highest_instance_id + 1

    def labels_exist(self, frame_num):
        """
        Returns true if track labels have already been assigned to a given frame
        :param frame_num: int. The frame ID (0, ..., T-1)
        :return:
        """
        return self._frame_labels[frame_num] is not None

    def has_fg_pixels(self, frame_num):
        assert self.labels_exist(frame_num)
        return self._frame_labels[frame_num].numel() > 0

    def get_labels(self, frame_nums):
        assert all(self.labels_exist(t) for t in frame_nums)
        return [self._frame_labels[t] for t in frame_nums]

    def update_labels(self, frame_num, labels):
        """
        Similar to add_labels, but is meant to be used when updating the labels for a given frame (e.g. using a
        long-range association measure). This method makes sure that the number of points in the previous and updated
        labels are the same.
        :param frame_num: int. The frame ID (0, ..., T-1)
        :param labels: tensor(N, E)
        :return:
        """
        assert self.labels_exist(frame_num)
        assert not self._is_frozen[frame_num]
        assert self._frame_labels[frame_num].shape == self._frame_labels[frame_num].shape
        self._frame_labels[frame_num] = labels
        if labels.numel() > 0:
            self._highest_instance_id = max(self._highest_instance_id, labels.max().item())
        return self._highest_instance_id

    def freeze_frame(self, frame_num):
        """
        Safety precaution: when you're finished processing a given frame, call this method and it will ensure that no
        changes are made to the predicted labels of that frame in the future.
        :param frame_num:
        :return:
        """
        assert self.labels_exist(frame_num)
        self._is_frozen[frame_num] = True

    def get_track_mask_idxes(self):
        """
        Returns 3 dicts. The first contains final list of track as a dict with keys being the frame numbers and values
        being tensors containing the track ID for each foreground pixel. Note that this is just a flattened list of
        labels and not the final masks.

        The second dict contains the number of pixels belonging to each track ID (useful
        for breaking ties between tracks when generating the final masks).

        The third dict contains the temporal lifetime of each track ID (also useful
        for breaking ties between tracks when generating the final masks).
        :return: dict, dict
        """
        instance_id_num_pts = defaultdict(lambda: 0)
        instance_id_lifetimes = defaultdict(lambda: [10000, -1])

        for frame_num, labels_per_frame in enumerate(self._frame_labels):
            for id in labels_per_frame.unique().tolist():
                instance_id_num_pts[id] += (labels_per_frame == id).long().sum().item()
                instance_id_lifetimes[id][0] = min(frame_num, instance_id_lifetimes[id][0])
                instance_id_lifetimes[id][1] = max(frame_num, instance_id_lifetimes[id][1])

        instance_id_lifetimes = {k: v[1] - v[0] for k, v in instance_id_lifetimes.items()}
        return self._frame_labels, instance_id_num_pts, instance_id_lifetimes


class OnlineChainer(object):
    OUTLIER_LABEL = -1

    def __init__(self, clusterer, embedding_resize_factor):
        self.clusterer = clusterer
        self.resize_scale = embedding_resize_factor

    @torch.no_grad()
    def resize_tensors(self, subseq):
        if self.resize_scale == 1.0:
            return

        def resize(x):
            x = x.unsqueeze(0)
            x = F.interpolate(x, scale_factor=(1.0, self.resize_scale, self.resize_scale), mode='trilinear',
                              align_corners=False)
            return x.squeeze(0)

        subseq["embeddings"] = resize(subseq["embeddings"])
        subseq["seediness"] = resize(subseq["seediness"])
        subseq["bandwidths"] = resize(subseq["bandwidths"])

    @torch.no_grad()
    def process(self, masks, subsequences, return_fg_embeddings=False):
        """
        Performs clustering/stitching of tracklets for a video containing T frames.
        :param masks: foreground masks as tensor(T, H, W)
        :param subsequences: list(dict). The list contains one entry per sub-sequence. There can be an arbitrary number
        sub-sequences. Each dict must contain a 'frames' key with a list of frames belonging to that sub-sequence, and
        an 'embedding' key with a tensor of shape (E, T_subseq, H, W) containing the embeddings for that sub-sequence.
        :param return_fg_embeddings: bool
        :return:
        """
        num_frames = masks.shape[0]

        # convert masks into a list of foreground indices
        mask_idxes = masks_to_coord_list(masks)

        # store labels for each sub-sequence in a list: this will be used for generating visualizations
        subseq_labels_list = []
        subseq_clustering_meta_info = []
        track_container = TrackContainer(num_frames)
        next_track_label = 1

        fg_embeddings = []

        print("Clustering subsequences...")
        for i in tqdm(range(len(subsequences))):
            subseq = subsequences[i]
            if isinstance(subseq['frames'], dict):
                subseq['frames'] = sorted(subseq['frames'].keys())  # bug in InferenceModel where dict is saved instead of list

            subseq_mask_idxes = [mask_idxes[t] for t in subseq['frames']]  # split the embeddings for different frames

            subseq['embeddings'] = subseq['embeddings'].cuda()
            subseq['bandwidths'] = subseq['bandwidths'].cuda()
            subseq['seediness'] = subseq['seediness'].cuda()

            self.resize_tensors(subseq)

            assert subseq['embeddings'].shape[-2:] == masks.shape[-2:], \
                "Size mismatch between embeddings {} and masks {}".format(subseq['embeddings'].shape, masks.shape)

            subseq_labels, subseq_fg_embeddings, meta_info = self.cluster_subsequence(
                subseq_mask_idxes, subseq['embeddings'], subseq['bandwidths'], subseq['seediness'], next_track_label,
                return_fg_embeddings)
            # print("Subseq labels: ", meta_info['instance_labels'])

            subseq_labels_list.append(subseq_labels)

            if return_fg_embeddings:
                fg_embeddings.append(subseq_fg_embeddings.cpu())

            if i == 0:
                # first sub-sequence; use this to initialize the tracks
                subseq_labels_cpu = [l.cpu() for l in subseq_labels]
                next_track_label = track_container.add_labels(subseq['frames'], subseq_labels_cpu)
                subseq_clustering_meta_info.append(meta_info)
                continue

            # associate tracks using the overlapping frames between the current and the previous sub-sequence
            previous_subseq = subsequences[i-1]
            overlapping_frames = sorted(list(set(subseq['frames']).intersection(set(previous_subseq['frames']))))

            # get the track labels already assigned to these overlapping frames from the clustering output of the
            # previous sub-sequence
            overlapping_frame_existing_labels = track_container.get_labels(overlapping_frames)

            # get the track labels assigned to these overlapping frames from the clustering output of the current
            # sub-sequence
            overlapping_frames_current_labels = [
                subseq_labels[i] for i, t in enumerate(subseq['frames']) if t in overlapping_frames
            ]

            # associate these labels
            associations, _, _, _, _ = self.associate_clusters(
                overlapping_frame_existing_labels, overlapping_frames_current_labels)

            # update labels in the current subseq clustering accordingly
            for j, t in enumerate(subseq['frames']):
                if t in overlapping_frames:
                    continue

                for associated_label, current_subseq_label in associations:
                    subseq_labels[j] = torch.where(
                        subseq_labels[j] == current_subseq_label, torch.tensor(associated_label).to(subseq_labels[j]), subseq_labels[j])

                # add the updated labels to container
                subseq_labels_cpu = [l.cpu() for l in subseq_labels]
                next_track_label = track_container.add_labels([t], [subseq_labels_cpu[j]])

            # update the meta-info dict as well
            for associated_label, current_subseq_label in associations:
                idx = meta_info['instance_labels'].index(current_subseq_label)
                meta_info['instance_labels'][idx] = associated_label

            subseq_clustering_meta_info.append(meta_info)

            # clear tensors (save RAM)
            subseq["embeddings"] = subseq["bandwidths"] = subseq["seediness"] = None

        return track_container.get_track_mask_idxes(), mask_idxes, subseq_labels_list, fg_embeddings, \
               subseq_clustering_meta_info

    def cluster_subsequence(self, mask_idxes, embeddings, bandwidths, seediness, label_start, return_fg_embeddings):
        """
        Performs clustering within a sub-sequence
        :param mask_idxes: list(T, tuple(tensor(M), tensor(M))
        :param embeddings: tensor(E, T, H, W)
        :param bandwidths: tensor(E, T, H, W) or None
        :param seediness: tensor(1, T, H, W) or None
        :param label_start: int
        :param return_fg_embeddings: bool
        :return:
        """
        assert len(mask_idxes) == embeddings.shape[1]

        # extract foreground embeddings
        embeddings = embeddings.permute(1, 2, 3, 0).unbind(0)  # list(T, tensor(H, W, E))

        bandwidths = bandwidths.permute(1, 2, 3, 0)

        bandwidths = bandwidths.unbind(0)  # list(T, tensor(H, W, E))
        seediness = seediness.permute(1, 2, 3, 0).unbind(0)  # list(T, tensor(H, W, 1))

        embeddings_flat, bandwidths_flat, seediness_flat, num_fg_embeddings = [], [], [], []

        for t, (mask_idxes_per_frame, embeddings_per_frame) in enumerate(zip(mask_idxes, embeddings)):
            embeddings_flat.append(embeddings_per_frame[mask_idxes_per_frame])
            num_fg_embeddings.append(mask_idxes_per_frame[0].numel())

            if bandwidths:
                bandwidths_flat.append(bandwidths[t][mask_idxes_per_frame])

            if seediness:
                seediness_flat.append(seediness[t][mask_idxes_per_frame])

        embeddings_flat = torch.cat(embeddings_flat)
        if bandwidths_flat:
            bandwidths_flat = torch.cat(bandwidths_flat)
        if seediness_flat:
            seediness_flat = torch.cat(seediness_flat)

        cluster_labels, clustering_meta_info = self.clusterer(
            embeddings_flat, bandwidths=bandwidths_flat, seediness=seediness_flat, cluster_label_start=label_start,
            return_label_masks=return_fg_embeddings)
        assert cluster_labels.numel() == embeddings_flat.shape[0]

        # split the labels by frame
        return list(cluster_labels.split(num_fg_embeddings, 0)), embeddings_flat, clustering_meta_info

    def associate_clusters(self, labels_1, labels_2):
        """
        Associates clusters and resolves inconsistencies for a pair of labels for a given frame.
        :param labels_1: list(tensor(N, E)).
        :param labels_2: list(tensor(N, E)).
        :return:
        """
        if not torch.is_tensor(labels_1):
            labels_1 = torch.cat(labels_1).cuda()

        if not torch.is_tensor(labels_2):
            labels_2 = torch.cat(labels_2).cuda()

        assert labels_1.shape == labels_2.shape, "Shape mismatch: {}, {}".format(labels_1.shape, labels_2.shape)

        # do not associate the outlier ID with anything
        unique_labels_1 = list(set(labels_1.unique().tolist()) - {self.OUTLIER_LABEL})
        unique_labels_2 = list(set(labels_2.unique().tolist()) - {self.OUTLIER_LABEL})

        assert not set(unique_labels_1).intersection(set(unique_labels_2)), \
            "Labels overlap: {}, {}".format(unique_labels_1, unique_labels_2)

        association_costs = np.zeros((len(unique_labels_1), len(unique_labels_2)), np.float32)
        recall_12 = np.zeros((len(unique_labels_1), len(unique_labels_2)), np.float32)

        # iterate over pairs of labels
        for i1, i2 in [(i1, i2) for i1 in range(len(unique_labels_1)) for i2 in range(len(unique_labels_2))]:
            l1, l2 = unique_labels_1[i1], unique_labels_2[i2]
            l1_active_pts = labels_1 == l1
            l2_active_pts = labels_2 == l2

            intersection = (l1_active_pts & l2_active_pts).float().sum()
            union = (l1_active_pts | l2_active_pts).float().sum()
            iou = intersection / union

            # print("IoU ({}, {}) = {}".format(l1, l2, iou.item()))
            association_costs[i1, i2] = 1. - iou.item()
            recall_12[i1, i2] = intersection / l1_active_pts.sum(dtype=torch.float32)

        idxes_1, idxes_2 = linear_sum_assignment(association_costs)

        associations = []
        unassigned_labels_1 = set(unique_labels_1)
        unassigned_labels_2 = set(unique_labels_2)

        for i1, i2 in zip(idxes_1, idxes_2):
            l1, l2 = unique_labels_1[i1], unique_labels_2[i2]
            associations.append((l1, l2))
            unassigned_labels_1.remove(l1)
            unassigned_labels_2.remove(l2)

        return associations, unassigned_labels_1, unassigned_labels_2, association_costs[idxes_1, idxes_2], \
               (recall_12, unique_labels_1, unique_labels_2)
