from collections import defaultdict
from stemseg.utils import ModelOutputConsts as ModelOutput

import torch


class ModelOutputManager(object):
    def __init__(self, division_factor, excluded_keys=()):
        self.division_factor = float(division_factor)

        self.tensor_vars = defaultdict(lambda: 0.)
        self.other_vars = defaultdict(lambda: 0.)

        self.excluded_keys = excluded_keys

    @torch.no_grad()
    def accumulate_vars(self, d):
        for k, v in d.items():
            if k in self.excluded_keys:
                continue

            if torch.is_tensor(v):
                self.tensor_vars[k] += (v.detach() / self.division_factor)
            else:
                self.other_vars[k] += (v / self.division_factor)

    def __call__(self, model_output):
        optimization_losses = model_output[ModelOutput.OPTIMIZATION_LOSSES]
        total_optimization_loss = sum(list(optimization_losses.values())) / self.division_factor

        self.accumulate_vars(model_output[ModelOutput.OPTIMIZATION_LOSSES])
        self.accumulate_vars(model_output[ModelOutput.OTHERS])

        return total_optimization_loss

    def reset(self):
        tensor_vars = dict(self.tensor_vars)
        other_vars = dict(self.other_vars)

        self.tensor_vars = defaultdict(lambda: 0.)
        self.other_vars = defaultdict(lambda: 0.)

        return tensor_vars, other_vars
