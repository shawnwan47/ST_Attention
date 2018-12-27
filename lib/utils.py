import datetime
from constants import EPS


def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def cat_strs(*args):
    args = [arg for arg in args if isinstance(arg, str)]
    return '_'.join(args)


def minute_str(minute):
    def time_str(t):
        return str(t) if t >= 10 else '0' + str(t)
    hour, minute = divmod(minute, 60)
    return time_str(hour) + ':' + time_str(minute)
