import sys
import torch.nn as nn
from torch.nn import functional as F
from utils.constants import EPS


class Error(object):
    """
    Accumulator for loss statistics.
    Calculating mae, rmse, mape, wape.
    """
    def __init__(self, mae=0, rmse=0, mape=0, wape=0, count=0):
        self.mae = mae
        self.rmse = rmse
        self.mape = mape
        self.wape = wape
        self.count = count

    def __repr__(self):
        return 'mae:{mae:.4f} rmse:{rmse:.4f} mape:{mape:.4f} wape:{wape:.4f}'.format(
            mae=self.mae / self.count,
            rmse=self.rmse / self.count,
            mape=self.mape / self.count,
            wape=self.wape / self.count
        )

    def update(self, stat):
        self.mae += stat.mae
        self.rmse += stat.rmse
        self.mape += stat.mape
        self.wape += stat.wape
        self.count += stat.count


class Rescaler(nn.Module):
    def __init__(self, mean, scale):
        assert len(mean) == len(scale)
        super().__init__()
        self.mean, self.scale = mean, scale

    def forward(self, input):
        return (input * scale) + mean


class Loss(nn.Module):
    def __init__(self, loss, mean, scale):
        assert loss in ('mae', 'rmse')
        super().__init__()
        self.loss = loss
        self.rescaler = Rescaler(mean, scale)


    def forward(self, input, target):
        input = self.rescaler(input)
        input, target = self.mask(input, target)
        mae = self._compute_mae(input, target)
        rmse = self._compute_rmse(input, target)
        mape = self._compute_mape(input, target)
        wape = self._compute_wape(input, target)
        return Error(mae, rmse, mape, wape, 1)

    @staticmethod
    def mask(input, target):
        mask = ~target.isnan(target)
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
