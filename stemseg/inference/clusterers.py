from collections import defaultdict
from time import time as current_time

import torch


class ClustererBase(object):
    def __init__(self):
        self._time_log = defaultdict(list)

    def __call__(self, embeddings, *args, **kwargs):
        assert embeddings.dtype == torch.float32

        start_time = current_time()
        output = self._process(embeddings, *args, **kwargs)
        duration = current_time() - start_time
        self._time_log[embeddings.shape[0]].append(duration)
        return output

    def _process(self, embeddings, *args, **kwargs):
        raise NotImplementedError("Must be implemented by derived class")

    def reset_time_log(self):
        self._time_log = defaultdict(list)

    @property
    def average_time(self):
        all_times = sum(list(self._time_log.values()), [])
        return sum(all_times) / float(len(all_times))

    name = property(fget=lambda self: self._name)


class SequentialClustering(ClustererBase):
    def __init__(self, primary_prob_thresh, secondary_prob_thresh, min_seediness_prob,
                 n_free_dims, free_dim_stds, device, max_instances=20):
        super().__init__()

        self.thresholding_mode = "probability"

        self.primary_prob_thresh = primary_prob_thresh
        self.secondary_prob_thresh = secondary_prob_thresh

        self.min_seediness_prob = min_seediness_prob

        self.max_instances = max_instances

        self.n_free_dims = n_free_dims
        self.free_dim_stds = free_dim_stds
        self.device = device

    @staticmethod
    def distances_to_prob(distances):
        return (-0.5 * distances).exp()

    @staticmethod
    def compute_distance(embeddings, center, bandwidth):
        return (torch.pow(embeddings - center, 2) * bandwidth).sum(dim=-1).sqrt()

    @torch.no_grad()
    def _process(self, embeddings, bandwidths, seediness, cluster_label_start=1, *args, **kwargs):
        if embeddings.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=embeddings.device), \
                   {
                        'instance_labels': [],
                        'instance_centers': [],
                        'instance_stds': [],
                        'instance_masks': []
                   }

        input_device = embeddings.device
        embeddings = embeddings.to(device=self.device)

        assert torch.is_tensor(bandwidths)
        if bandwidths.shape[0] != embeddings.shape[0]:
            bandwidths = bandwidths.expand_as(embeddings)

        bandwidths = bandwidths.to(device=self.device)

        if self.n_free_dims == 0:
            assert embeddings.shape == bandwidths.shape

        assert torch.is_tensor(seediness)
        seediness.shape[0] == embeddings.shape[0], "Seediness shape: {}, embeddings shape: {}".format(
            seediness.shape, embeddings.shape)
        seediness = seediness.squeeze(1).to(device=self.device)  # [N, 1] -> [N]

        label_masks = []
        unique_labels = []
        label_centers = []
        label_stds = []

        return_label_masks = kwargs.get("return_label_masks", False)

        total_points = embeddings.shape[0]
        labels = torch.full((total_points,), -1, dtype=torch.long, device=embeddings.device)
        label_distances = []
        num_unassigned_pts = total_points

        if self.n_free_dims > 0:
            free_dim_stds = torch.tensor(self.free_dim_stds).to(embeddings)
            free_dim_bandwidths = 1. / (free_dim_stds ** 2)
        else:
            free_dim_stds, free_dim_bandwidths = torch.zeros(0).to(embeddings), torch.zeros(0).to(embeddings)

        for i in range(self.max_instances):
            available_embeddings_mask = labels == -1
            num_unassigned_pts = available_embeddings_mask.sum(dtype=torch.long)
            if num_unassigned_pts == 0:
                break

            next_center, bandwidth, prob = self._get_next_instance_center(
                embeddings[available_embeddings_mask], bandwidths[available_embeddings_mask],
                seediness[available_embeddings_mask])

            if prob < self.min_seediness_prob:
                break

            bandwidth = torch.cat((bandwidth, free_dim_bandwidths), 0)

            instance_label = i + cluster_label_start

            unique_labels.append(instance_label)
            label_centers.append(next_center.tolist())
            label_stds.append((1. / bandwidth).clamp(min=1e-8).sqrt().tolist())

            # compute probability for all embedding under this center and bandwidth
            distances = torch.full_like(labels, 1e8, dtype=torch.float32, device=embeddings.device)
            distances[available_embeddings_mask] = self.compute_distance(
                embeddings[available_embeddings_mask], next_center, bandwidth)

            # store in dict for later use
            label_distances.append(distances)

            # update labels for all embeddings which are unassigned and fall within the primary prob threshold
            probs = torch.zeros_like(distances)
            probs[available_embeddings_mask] = self.distances_to_prob(distances[available_embeddings_mask])
            match_mask = (probs > self.primary_prob_thresh) & available_embeddings_mask

            # else:
            #     raise ValueError("Should not be here")

            labels = torch.where(match_mask, torch.tensor(instance_label, device=self.device), labels)

            if return_label_masks:
                label_masks.append(match_mask.cpu())

        # perform secondary assignment for unassigned points
        if num_unassigned_pts > 0 and label_distances:
            label_distances = torch.stack(label_distances, dim=1)  # [E, N]  (N = number of clusters)

            # find ID of cluster to which each point has the highest probability of belonging
            min_distance, min_distance_label = label_distances.max(dim=1)
            min_distance_label += cluster_label_start

            probs = self.distances_to_prob(min_distance)
            update_mask = (probs > self.secondary_prob_thresh) & available_embeddings_mask

            labels = torch.where(update_mask, min_distance_label, labels)

        return labels.to(input_device), {
            'instance_labels': unique_labels,
            'instance_centers': label_centers,
            'instance_stds': label_stds,
            'instance_masks': label_masks
        }

    def _get_next_instance_center(self, embeddings, bandwidths, seediness):
        if self.n_free_dims == 0:
            assert embeddings.shape == bandwidths.shape
        assert embeddings.numel() > 0
        assert embeddings.shape[0] == seediness.shape[0]

        max_prob_idx = seediness.argmax()
        return embeddings[max_prob_idx], bandwidths[max_prob_idx], seediness[max_prob_idx]
