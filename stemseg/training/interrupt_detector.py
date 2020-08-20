from argparse import Namespace, ArgumentParser

import signal


class InterruptException(RuntimeError):
    def __init__(self, *args):
        super(self.__class__, self).__init__(*args)


class InterruptDetector:
    def __init__(self):
        self.__is_interrupted = False

    def start(self):
        signal.signal(signal.SIGINT, self.__set_interrupted)
        signal.signal(signal.SIGTERM, self.__set_interrupted)

    def __set_interrupted(self, signum, frame):
        self.__is_interrupted = True

    is_interrupted = property(fget=lambda self: self.__is_interrupted)


def parse_args(parser):
    assert isinstance(parser, ArgumentParser)
    args = parser.parse_args()

    pos_group, optional_group = parser._action_groups[0], parser._action_groups[1]
    args_dict = args._get_kwargs()
    pos_optional_arg_names = [arg.dest for arg in pos_group._group_actions] + [arg.dest for arg in optional_group._group_actions]
    pos_optional_args = {name: value for name, value in args_dict if name in pos_optional_arg_names}
    other_group_args = dict()

    if len(parser._action_groups) > 2:
        for group in parser._action_groups[2:]:
            group_arg_names = [arg.dest for arg in group._group_actions]
            other_group_args[group.title] = Namespace(**{name: value for name, value in args_dict if name in group_arg_names})

    combined_args = pos_optional_args
    combined_args.update(other_group_args)
    return Namespace(**combined_args)