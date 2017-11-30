import torch
from Constants import EPS


def Denormalize(data_norm, data_mean, data_std):
    return data_norm * data_std + data_mean


def WAPE(outputs, targets):
    loss = torch.abs(targets - outputs)
    return 2 * loss.sum(1).div((targets + outputs).sum(1) + EPS).mean()


def MAPE(outputs, targets):
    loss = torch.abs(targets - outputs)
    return 2 * loss.div(targets + outputs + EPS).mean()
