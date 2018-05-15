import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from constants import EPS


class Metric(object):
    def __init__(self, metric):
        self.metric = metric
        self.count = 1

    def __str__(self):
        return str(self.metric / self.count)

    def update(self, other):
        self.metric += other.metric
        self.count += other.count


class MultiMetrics(object):
    def __init__(self):
        self.metrics = None

    def update(self, metrics):
        if self.metrics is None:
            self.metrics = metrics
        else:
            for error1, error2 in zip(self.metrics, metrics):
                error1.update(error2)

    def __str__(self):
        return '  '.join([str(error) for error in self.metrics])


class Loss(nn.Module):
    def __init__(self, loss, metric, futures, mean, std):
        assert loss in ('mae', 'rmse')
        assert metric in ('mape', 'wape')
        super().__init__()
        self.loss = loss
        self.metric = metric
        self.futures = futures
        self.mean, self.std = mean, std

    def forward(self, input, target):
        '''
        input, target: batch x len x num
        loss: masked reduced loss
        metric: multi-step metric
        '''
        input = self._rescale(input)
        loss = self._compute_loss(input, target)
        metric = [self._compute_metric(input[:, i], target[:, i])
                  for i in self.futures]
        return loss, metric

    def _rescale(self, input):
        return (input * (self.std + EPS)) + self.mean

    def _compute_loss(self, input, target):
        input, target = self._mask_select(input, target)
        if self.loss == 'mae':
            return self._compute_mae(input, target)
        elif self.loss == 'rmse':
            return self._compute_rmse(input, target)

    def _compute_metric(self, input, target):
        input, target = self._mask_select(input, target)
        if self.metric == 'mape':
            metric = self._compute_mape(input, target)
        elif self.metric == 'wape':
            metric = self._compute_wape(input, target)
        return Metric(metric.data[0])

    @staticmethod
    def _mask_select(input, target):
        mask = ~torch.isnan(target)
        return input.masked_select(mask), target.masked_select(mask)

    @staticmethod
    def _compute_mae(input, target):
        return F.l1_loss(input, target)

    @staticmethod
    def _compute_rmse(input, target):
        return F.mse_loss(input, target).sqrt()

    @staticmethod
    def _compute_mape(input, target):
        return ((input - target).abs() / (target + EPS)).mean()

    @staticmethod
    def _compute_wape(input, target):
        return F.l1_loss(input, target) / target.mean()
