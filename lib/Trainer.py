import numpy as np
from torch.nn.utils import clip_grad_norm_

from constants import EPS
from lib import Loss
from lib import pt_utils


class Rescaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input * (self.std + EPS)) + self.mean


class Trainer:
    def __init__(self, model, rescaler, criterion, loss,
                 optimizer, scheduler, epoches, iterations, cuda):
        self.model = model
        self.rescaler = rescaler
        self.criterion = criterion
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cuda = cuda
        self.epoches = epoches
        self.teach = 1
        self.teach_annealing = 0.01 ** (1 / epoches)

    def train(self, dataloader):
        metrics = Loss.MetricDict()
        for iter in range(self.iterations):
            data, time, day, target = next(dataloader)
            if self.cuda:
                data = data.cuda()
                time = time.cuda()
                day = day.cuda()
                target = target.cuda()
            output = self.model(data, time, day, self.teach)
            if isinstance(output, tuple):
                output = output[0]
            output = self.rescaler(output)
            metric = self.loss(output, target)
            output, target = pt_utils.mask_target(output, target)
            crit = self.criterion(output, target)
            self.optimizer.zero_grad()
            crit.backward()
            clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            del output
            metrics = metrics + metric
        return metrics

    def eval(self, dataloader):
        metrics = Loss.MetricDict()
        infos = []
        for data, time, day, target in dataloader:
            if self.cuda:
                data = data.cuda()
                time = time.cuda()
                day = day.cuda()
                target = target.cuda()
            output = self.model(data, time, day)
            if isinstance(output, tuple):
                output, info = output[0], output[1:]
                infos.append([pt_utils.torch_to_numpy(i) for i in info])
                del info
            output = self.rescaler(output)
            metrics = metrics + self.loss(output, target)
            del output
        if infos:
            infos = [np.concatenate(info) for info in zip(*infos)]
        return metrics, infos

    def run(self, data_train, data_valid, data_test):
        for epoch in range(self.epoches):
            error_train = self.train(data_train)
            error_valid, _ = self.eval(data_valid)
            error_test, _ = self.eval(data_test)
            print(f'Epoch: {epoch}',
                  f'train: {error_train}',
                  f'valid: {error_valid}',
                  f'test:  {error_test}',
                  f'teach ratio: {self.teach}',
                  f'learning rate: {self.optimizer.param_groups[0]["lr"]}',
                  sep='\n')
            self.scheduler.step()
            self.teach *= self.teach_annealing
