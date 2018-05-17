import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import EPS
from lib.pt_utils import mask_target


class Metric:
    def __init__(self, value, num):
        self.value = value
        self.num = num

    def __str__(self):
        if isinstance(self.value, float):
            return '0.'
        else:
            return f'{self.value:.3f}'

    def __add__(self, other):
        num = self.num + other.num
        value = (self.value * self.num + other.value * other.num) / num
        return Metric(value, num)


class Metrics:
    def __init__(self, keys, values, num):
        assert len(keys) == len(values)
        self.keys = keys
        self.values = values
        self.num = num

    def __repr__(self):
        return ' '.join([f'{metric}:{value:.2f}'
                         for metric, value in zip(self.keys, self.values)])

    def __add__(self, other):
        assert self.keys == other.keys
        n1, n2 = self.num, other.num
        values = [(v1 * n1 + v2 * n2) / (n1 + n2)
                  for v1, v2 in zip(self.values, other.values)]
        return Metrics(self.keys, values, n1 + n2)


class MetricList:
    def __init__(self, metrics=None):
        self.metrics = metrics

    def __len__(self):
        if self.metrics is None:
            return None
        else:
            return len(self.metrics)

    def __add__(self, other):
        if self.metrics is None:
            return other
        elif other.metrics is None:
            return self
        else:
            assert(len(self) == len(other))
            return MetricList([m1 + m2
                               for m1, m2 in zip(self.metrics, other.metrics)])

    def __str__(self):
        if self.metrics is None:
            return None
        else:
            return '\t'.join([str(metric) for metric in self.metrics])


class Loss:
    def __init__(self, metrics, futures):
        self.metrics = metrics
        self.futures = futures

    def __call__(self, output, target):
        return MetricList([
            self._eval(output[:, future], target[:, future])
            for future in self.futures])

    def _eval(self, output, target):
        output, target = mask_target(output, target)
        if not target.numel():
            return Metrics(self.metrics, [0] * len(self.metrics), 0)
        else:
            values = [self._compute_loss(output, target, metric)
                      for metric in self.metrics]
            return Metrics(self.metrics, values, target.numel())

    @staticmethod
    def _compute_loss(output, target, loss):
        if loss == 'mae':
            loss = F.l1_loss(output, target)
        elif loss == 'rmse':
            loss = F.mse_loss(output, target).sqrt()
        elif loss == 'mape':
            loss = ((output - target).abs() / (target + EPS)).mean() * 100
        elif loss == 'wape':
            loss = F.l1_loss(output, target) / (target.mean() + EPS) * 100
        return loss.item()
