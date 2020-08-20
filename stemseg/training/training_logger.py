from time import time as current_time

import os
import tensorboardX


class TrainingLogger(object):
    def __init__(self, output_dir, num_iterations=None):
        self.total_iterations = num_iterations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.__writer = tensorboardX.SummaryWriter(self.output_dir)
        self.__train_start_time = None
        self.__latest_timestamp = None
        self.__latest_iteration_num = None
        self.__pause_duration = 0.

    @property
    def elapsed_time(self):
        return self.__latest_timestamp - self.__train_start_time

    def start_timer(self):
        if self.__train_start_time is not None and self.__latest_iteration_num is not None:
            self.__pause_duration += (current_time() - self.__latest_timestamp)
        else:
            self.__train_start_time = current_time()

    def add_training_point(self, iteration_num, add_to_summary, **kwargs):
        self.__latest_timestamp = current_time()
        self.__latest_iteration_num = iteration_num
        if add_to_summary:
            for scalar_name, value in kwargs.items():
                self.__writer.add_scalar('training_' + scalar_name, value, iteration_num)

    def add_validation_run_results(self, num_training_iterations, **kwargs):
        for scalar_name, value in kwargs.items():
            self.__writer.add_scalar('validation_' + scalar_name, value, num_training_iterations)

    def compute_eta(self, as_string=True):
        assert self.__train_start_time is not None and self.__latest_timestamp is not None
        avg_time_per_iter = (self.__latest_timestamp - self.__train_start_time - self.__pause_duration) / float(self.__latest_iteration_num)
        eta = float(self.total_iterations - self.__latest_iteration_num) * avg_time_per_iter
        if not as_string:
            return eta, avg_time_per_iter

        days, rem = divmod(eta, 3600*24)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:d}-{:02d}:{:02d}:{:02d}".format(int(days), int(hours), int(minutes), int(seconds)), avg_time_per_iter

    def state_dict(self):
        return {'total_iterations': self.total_iterations,
                '_' + self.__class__.__name__ + '__train_start_time': self.__train_start_time,
                '_' + self.__class__.__name__ + '__latest_timestamp': self.__latest_timestamp,
                '_' + self.__class__.__name__ + '__latest_iteration_num': self.__latest_iteration_num,
                '_' + self.__class__.__name__ + '__pause_duration': self.__pause_duration}

    def load_state_dict(self, d):
        for key in d:
            assert key in self.__dict__, "Invalid parameter '%s' in state dict" % key
        self.__dict__.update(d)
