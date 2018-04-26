import sys
import torch.nn as nn
from torch.nn import functional as F
from utils.constants import eps


class Statistics(object):
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
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        return self.loss / self.n_words

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; xent: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.xent(),
               time.time() - start))
        sys.stdout.flush()


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
        self._loss = loss
        self.rescaler = Rescaler(mean, scale)


    def forward(self, input, target):
        input = self.rescaler(input)
        target = self.rescaler(target)
        mae = self._compute_mae(input, target)
        rmse = self._compute_rmse(input, target)
        mape = self._compute_mape(input, target)
        wape = self._compute_wape(input, target)
        error = Error(mae, rmse, mape, wape)
        return getattr(error, self._loss), error

    @staticmethod
    def _compute_mae(input, target):
        return F.l1_loss(input, target)

    @staticmethod
    def _compute_rmse(input, target):
        return F.mse_loss(input, target).sqrt()

    @staticmethod
    def _compute_mape(input, target):
        return ((input - target).abs() / (target + eps)).mean()

    @staticmethod
    def _compute_wape(input, target):
        return F.l1_loss(input, target) / target.mean()
