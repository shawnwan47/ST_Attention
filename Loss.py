import torch
from Consts import EPS


def WAPE(outputs, targets):
    return torch.abs(targets - outputs).sum() / targets.sum()


def MAPE(outputs, targets):
    loss = torch.abs(targets - outputs)
    return 2 * loss.div(targets + outputs + EPS).mean()
