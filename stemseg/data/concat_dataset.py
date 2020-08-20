from torch.utils.data import Dataset

import math
import random
import torch


class SparseDataset(Dataset):
    def __init__(self, dataset, num_samples):
        assert num_samples < len(dataset), "SparseDataset is only applicable when num_samples < len(dataset)"
        self.dataset = dataset
        self.num_samples = num_samples

        random.seed(42)
        self.idxes = list(range(len(dataset)))
        random.shuffle(self.idxes)
        self.idxes = self.idxes[:len(dataset)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.dataset[self.idxes[index]]


class ConcatDataset(Dataset):
    def __init__(self, datasets, total_samples, weights=None):
        if weights is None:
            weights = [1. / float(len(datasets)) for _ in range(len(datasets))]

        assert abs(sum(weights) - 1.) < 1e-6, "Sum of weights is {}. Should be 1".format(sum(weights))

        self.id_mapping = []
        self.samples_per_dataset = []
        for i, (wt, ds) in enumerate(zip(weights, datasets)):
            assert 0. < wt <= 1.
            num_samples_ds = int(round(wt * total_samples))
            if num_samples_ds < len(ds):
                ds = SparseDataset(ds, num_samples_ds)

            repetitions = int(math.floor(num_samples_ds / float(len(ds))))
            idxes = sum([list(range(len(ds))) for _ in range(repetitions)], [])

            rem_idxes = torch.linspace(0, len(ds)-1, num_samples_ds - len(idxes)).round().long().tolist()
            idxes += rem_idxes

            self.id_mapping.extend([(i, j) for j in idxes])
            self.samples_per_dataset.append(num_samples_ds)

        self.datasets = datasets
        self.weights = weights

        assert len(self.id_mapping) == total_samples

    def __len__(self):
        return len(self.id_mapping)

    def __getitem__(self, index):
        ds_idx, sample_idx = self.id_mapping[index]
        return self.datasets[ds_idx][sample_idx]
