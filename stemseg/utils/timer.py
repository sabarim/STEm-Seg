from time import time as current_time


class Timer(object):
    _TIMERS = dict()

    def __init__(self, name):
        self._name = name
        self.__tic_time = None
        self.__total_duration = 0.0

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()

    def tic(self):
        assert self.__tic_time is None, "tic() has already been called for timer '{}'".format(self._name)
        self.__tic_time = current_time()

    def toc(self):
        assert self.__tic_time is not None, "tic() has not been called for timer '{}'".format(self._name)
        self.__total_duration += current_time() - self.__tic_time
        self.__tic_time = None

    @property
    def running(self):
        return self.__tic_time is not None

    @property
    def paused(self):
        return not self.running

    @property
    def total_duration(self):
        return self.__total_duration

    @staticmethod
    def create(name):
        assert name not in Timer._TIMERS, "Timer with name '{}' already exists".format(name)
        timer = Timer(name)
        Timer._TIMERS[name] = timer
        return timer

    @staticmethod
    def get(name):
        if name not in Timer._TIMERS:
            return Timer.create(name)
        else:
            return Timer._TIMERS[name]

    @staticmethod
    def get_duration(name):
        timer = Timer._TIMERS.get(name, None)
        assert timer is not None, "No timer named '{}' exists".format(name)
        return timer.total_duration

    @staticmethod
    def get_durations_sum():
        return sum([timer.total_duration for timer in Timer._TIMERS.values()])

    @staticmethod
    def print_durations():
        durations_sum = 0.
        for name, timer in Timer._TIMERS.items():
            print(" - {}: {:03f} sec".format(name, timer.total_duration))
            durations_sum += timer.total_duration
        print(" - TOTAL: {:03f} sec".format(durations_sum))

    @staticmethod
    def log_duration(*timer_names):
        def wrap(f):
            def wrap2(*args, **kwargs):
                timers_to_pause = []

                for name in timer_names:
                    timer = Timer.get(name)
                    if timer.paused:
                        timers_to_pause.append(timer)
                        timer.tic()

                output = f(*args, **kwargs)

                for timer in timers_to_pause:
                    timer.toc()

                return output

            return wrap2
        return wrap

    @staticmethod
    def exclude_duration(*timer_names):
        def wrap(f):
            def wrap2(*args, **kwargs):
                timers_to_resume = []

                for name in timer_names:
                    if name in Timer._TIMERS:
                        timer = Timer.get(name)
                        if timer.running:
                            timer.toc()
                            timers_to_resume.append(timer)

                output = f(*args, **kwargs)

                for timer in timers_to_resume:
                    timer.tic()

                return output

            return wrap2
        return wrap

    # @staticmethod
    # def exclude_duration(name=''):
    #     def wrap(f):
    #         def wrap2(*args, **kwargs):
    #             if name in Timer._TIMERS:
    #                 timer = Timer.get(name)
    #                 paused = timer.running
    #                 if paused:
    #                     timer.toc()
    #
    #                 output = f(*args, **kwargs)
    #
    #                 if paused:
    #                     timer.tic()
    #
    #             else:
    #                 output = f(*args, **kwargs)
    #
    #             return output
    #
    #         return wrap2
    #     return wrap
