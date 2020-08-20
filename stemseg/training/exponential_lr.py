import math
import torch


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, decay_factor, decay_steps, start_at=0, last_epoch=-1):
        assert decay_steps > 0
        assert decay_factor < 1.0
        self.gamma = math.exp(math.log(decay_factor) / float(decay_steps))
        self.start_at = start_at

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_at:
            return self.base_lrs

        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
