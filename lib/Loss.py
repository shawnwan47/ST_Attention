import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import EPS
from lib.pt_utils import mask_target


class Metric:
    def __init__(self, value, norm):
        self.value = value
        self.norm = norm

    def __repr__(self):
        return f'{self.key}:{self.value:.2f}'

    def __add__(self, other):
        norm = self.norm + other.norm
        value = (self.value * self.norm + other.value * other.norm) / norm
        return Metric(value, norm)


class MetricList(dict):
    def __add__(self, other):
        if not self:
            return other
        elif not other:
            return self
        else:
            assert(len(self) == len(other))
            return MetricList({key: self[key] + other[key]
                               for key in self.keys()})

    def __iadd__(self, other):
        if not self:
            return other
        elif not other:
            return self
        else:
            assert(len(self) == len(other))
            for key in self.keys():
                self[key] += other[key]
        return self

    def __repr__(self):
        return ' '.join([f'{key}:{val}' for key, val in self.items()])


class Loss:
    def __init__(self, metrics, horizons):
        self.metrics = metrics
        self.horizons = horizons

    def __call__(self, output, target):
        ret = MetricList({horizon: self._eval(output[:, horizon], target[:, horizon])
                          for horizon in self.horizons})
        ret['average'] = self._eval(output, target)
        return ret

    def _eval(self, output, target):
        output, target = mask_target(output, target)
        if not target.numel():
            return MetricList()
        else:
            metrics = MetricList({metric: self._loss(output, target, metric)
                                  for metric in self.metrics]
            return metrics

    @staticmethod
    def _loss(output, target, loss):
        if loss == 'mae':
            loss = F.l1_loss(output, target)
        elif loss == 'rmse':
            loss = F.mse_loss(output, target).sqrt()
        elif loss == 'mape':
            loss = ((output - target).abs() / (target + EPS)).mean() * 100
        elif loss == 'wape':
            loss = F.l1_loss(output, target) / (target.mean() + EPS) * 100

        if loss == 'wape':
            loss = Metric(loss.item(), target.mean() + EPS)
        else:
            loss = Metric(loss.item(), target.numel())
        return loss
