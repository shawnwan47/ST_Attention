

def aeq(*args):
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def od_distance(od_p, od_q):
    m, n = od_p.shape
    kl = 0
    for i in range(m):
        kl += entropy(od_p[i], od_q[i]) * od_p[i].sum()
    kl /= od_p.sum()
    return kl
