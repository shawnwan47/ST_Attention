import torch
from Constants import EPS


def getLoss(flow_real, flow_pred, flow_mean, flow_std):
    flow_real = torch.bmm(flow_real, flow_std) + flow_mean
    flow_pred = torch.bmm(flow_pred, flow_std) + flow_mean
    return torch.abs(flow_pred - flow_real)


def WAPE(flow_real, flow_pred, flow_mean, flow_std):
    loss = getLoss(flow_real, flow_pred, flow_mean, flow_std)
    return loss.sum(1).div(flow_real.sum(1))


def MAPE(flow_real, flow_pred, flow_mean, flow_std):
    loss = getLoss(flow_real, flow_pred, flow_mean, flow_std)
    return 2 * loss.div(EPS + flow_mean + flow_pred)
