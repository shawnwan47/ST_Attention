import sys
import torch
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

    def update(self, error):
        self.mae += error.mae
        self.rmse += error.rmse
        self.mape += error.mape
        self.wape += error.wape
        self.count += error.count


class MultiError(object):
    def __init__(self, errors):
        self.errors = errors

    def update(self, errors):
        for error1, error2 in zip(self.errors, errors):
            error1.update(error2)

    def __repr__(self):
        return '\t'.join(self.errors)


class Loss(nn.Module):
    def __init__(self, loss, futures, mean, std):
        assert loss in ('mae', 'rmse')
        super().__init__()
        self.loss = loss
        self.futures = futures
        self.mean, self.std = mean, std

    def forward(self, input, target):
        '''
        input, target: batch x len x num
        loss: masked reduced loss
        errors: multi-step errors
        '''
        input = self.rescale(input)
        loss = getattr(self._compute_error(input, target), self.loss)
        errors = [self._compute_error(input[:, i], target[:, i])
                  for i in self.futures]
        return loss, errors

    def rescale(self, input):
        return (input * (self.std + EPS)) + self.mean

    @staticmethod
    def _compute_loss(input, target):
        input, target = self._mask_select(input, target)
        if self.loss == 'mae':
            return self._compute_mae(input, target)
        elif self.loss == 'rmse':
            return self._compute_rmse(input, target)

    @staticmethod
    def _compute_error(input, target):
        input, target = self._mask_select(input, target)
        mae = self._compute_mae(input, target)
        rmse = self._compute_rmse(input, target)
        mape = self._compute_mape(input, target)
        wape = self._compute_wape(input, target)
        return Error(mae, rmse, mape, wape, 1)

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
