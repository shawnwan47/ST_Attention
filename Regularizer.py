import numpy as np
import torch


def Frobenius(mat):
    ret = (torch.sum(torch.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
    return torch.sum(ret) / mat.size(0)


def orthogonal(A):
    '''A: batch x model x weight'''
    assert A.dim() <= 3
    if A.dim() == 2:
        A = A.unsqueeze(0)
    AT = A.transpose(1, 2).contiguous()
    length = A.size(1)
    I = Variable(torch.from_numpy(np.eye(length)))
    return Frobenius(torch.bmm(A, AT) - I)


def continual(weight):
    '''
    weight: batch x length x dim
    '''
    pass
