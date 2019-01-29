import torch
import torch.nn as nn


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
        if metric == 'wape':
            loss = (output - target).abs().sum().item() * 100
            return Metric(loss, target.sum().item())
        elif metric == 'mae':
            loss = (output - target).abs().sum()
        elif metric == 'rmse':
            loss = (output - target).pow(2).sqrt().sum()
        elif metric == 'mape':
            loss = ((output - target).abs() / (target + 1e-8)).sum() * 100
        return Metric(loss.item(), target.numel())


class Metric:
    def __init__(self, loss, norm):
        self.loss = loss
        self.norm = norm

    def __float__(self):
        return self.loss / (self.norm + 1e-8)

    def __repr__(self):
        return f'{float(self):.2f}'

    def __iadd__(self, other):
        self.loss += other.loss
        self.norm += other.norm
        return self

    def __add__(self, other):
        loss = self.loss + other.loss
        norm = self.norm + other.norm
        return Metric(loss, norm)


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
