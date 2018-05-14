def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def cat_strs(*args):
    args = [arg for arg in args if isinstance(arg, str)]
    return '_'.join(args)
