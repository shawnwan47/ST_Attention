import torch
from Constants import EPS


def Denormalize(flow_norm, flow_mean, flow_std):
    return flow_norm * flow_std + flow_mean


def WAPE(flow_real, flow_pred, flow_mean, flow_std):
    flow_real = flow_real * flow_std + flow_mean
    flow_pred = flow_pred * flow_std + flow_mean
    loss = torch.abs(flow_real - flow_pred)
    return loss.sum(1).div(flow_real.sum(1) + EPS).mean()


def MAPE(flow_real, flow_pred, flow_mean, flow_std):
    flow_real = flow_real * flow_std + flow_mean
    flow_pred = flow_pred * flow_std + flow_mean
    loss = torch.abs(flow_real - flow_pred)
    return 2 * loss.div(flow_real + flow_pred + EPS).mean()
