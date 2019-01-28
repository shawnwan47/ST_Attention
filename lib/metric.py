import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import EPS


def mask_target(output, target):
    mask = ~torch.isnan(target)
    return output.masked_select(mask), target.masked_select(mask)


class TimeSeriesLoss:
    def __init__(self, metrics, horizons):
        self.metrics = metrics
        self.horizons = horizons

    def __call__(self, output, target):
        ret = MetricDict({horizon: self.eval(output[:, horizon], target[:, horizon])
                          for horizon in self.horizons})
        ret['avg'] = self.eval(output, target)
        return ret

    def get_criterion(self, lossdict):
        return float(lossdict['avg'][self.metrics[-1]])

    def eval(self, output, target):
        output, target = mask_target(output, target)
        if not target.numel():
            return MetricDict()
        else:
            return MetricDict({metric: self.calc_metric(output, target, metric)
                               for metric in self.metrics})

    @staticmethod
    def calc_metric(output, target, metric):
        if metric == 'mae':
            loss = F.l1_loss(output, target)
        elif metric == 'rmse':
            loss = F.mse_loss(output, target).sqrt()
        elif metric == 'mape':
            loss = ((output - target).abs() / (target + EPS)).mean() * 100
        elif metric == 'wape':
            loss = (output - target).abs().mean() / (target.mean() + EPS) * 100

        if metric == 'wape':
            return Metric(loss.item(), target.mean().item() + EPS)
        else:
            return Metric(loss.item(), target.numel())


class Metric:
    def __init__(self, value, norm):
        self.value = value
        self.norm = norm

    def __float__(self):
        return self.value

    def __repr__(self):
        return f'{self.value:.2f}'

    def __iadd__(self, other):
        self.value = (self.value * self.norm + other.value * other.norm)
        self.norm += other.norm
        self.value /= self.norm
        return self

    def __add__(self, other):
        norm = self.norm + other.norm
        value = (self.value * self.norm + other.value * other.norm) / norm
        return Metric(value, norm)


class MetricDict(dict):
    def __add__(self, other):
        if not (self and other):
            return self or other
        assert(self.keys() == other.keys())
        return MetricDict({key: self[key] + other[key]
                           for key in self.keys()})

    def __iadd__(self, other):
        if not (self and other):
            return self or other
        assert self.keys() == other.keys()
        for key in self.keys():
            self[key] += other[key]
        return self

    def __repr__(self):
        return ' '.join([f'{key}:{val}' for key, val in self.items()])
