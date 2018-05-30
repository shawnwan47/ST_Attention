import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import EPS
from lib.pt_utils import mask_target


class Metric:
    def __init__(self, key, value, num):
        self.key = key
        self.value = value
        self.num = num

    def __repr__(self):
        return f'{self.key}:{self.value:.2f}'

    def __add__(self, other):
        num = self.num + other.num
        value = (self.value * self.num + other.value * other.num) / num
        return Metric(self.key, value, num)


class MetricList(list):
    def __add__(self, other):
        if not self:
            return other
        elif not other:
            return self
        else:
            assert(len(self) == len(other))
            return MetricList([m1 + m2
                               for m1, m2 in zip(self, other)])

    def __iadd__(self, other):
        if not self:
            return other
        elif not other:
            return self
        else:
            assert(len(self) == len(other))
            for idx, metric in enumerate(other):
                self[idx] += metric
        return self

    def __str__(self):
        if not self:
            return None
        else:
            return ' '.join([metric for metric in self])


class Loss:
    def __init__(self, metrics, horizons):
        self.metrics = metrics
        self.horizons = horizons

    def __call__(self, output, target):
        ret = MetricList([
            self._eval(output[:, horizon], target[:, horizon])
            for horizon in self.horizons])
        ret.append(self._eval(output, target))
        return ret

    def _eval(self, output, target):
        output, target = mask_target(output, target)
        num = target.numel()
        if not num:
            return MetricList()
        else:
            metrics = [Metric(metric, self._loss(output, target, metric), num)
                       for metric in self.metrics]
            return MetricList(metrics)

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
        return loss.item()
