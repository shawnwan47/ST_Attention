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

    def get_criterion(self, lossdict):
        return float(lossdict['avg'][self.metrics[-1]])

    def __call__(self, output, target):
        ret = MetricDict({horizon: self._eval(output[:, horizon], target[:, horizon])
                          for horizon in self.horizons})
        ret['avg'] = self._eval(output, target)
        return ret

    def _eval(self, output, target):
        output, target = mask_target(output, target)
        if not target.numel():
            return MetricDict()
        else:
            return MetricDict({metric: self.get_loss(output, target, metric)
                               for metric in self.metrics})

    @staticmethod
    def get_loss(output, target, metric):
        if metric == 'mae':
            loss = F.l1_loss(output, target)
        elif metric == 'rmse':
            loss = F.mse_loss(output, target).sqrt()
        elif metric == 'mape':
            loss = ((output - target).abs() / (target + EPS)).mean() * 100
        elif metric == 'wape':
            loss = F.l1_loss(output, target) / (target.mean() + EPS) * 100

        if metric == 'wape':
            loss = Metric(loss.item(), target.mean() + EPS)
        else:
            loss = Metric(loss.item(), target.numel())
        return loss


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
