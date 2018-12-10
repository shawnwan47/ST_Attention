import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from lib import Loss
from lib import pt_utils


class Trainer:
    def __init__(self, model, framework, rescaler, criterion, loss,
                 optimizer, epoches, iterations, cuda):
        self.model = model
        self.framework = framework
        self.rescaler = rescaler
        self.criterion = criterion
        self.loss = loss
        self.optimizer = optimizer
        self.epoches = epoches
        self.iterations = iterations
        if cuda:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor
        else:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor

    def run_epoch(self, dataloader, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        error = Loss.MetricDict()
        for iter, (data, time, day, target) in enumerate(dataloader):
            if train and iter == self.iterations:
                break
            data = data.type(self.float_type)
            time = time.type(self.long_type)
            day = day.type(self.long_type)
            target = target.type(self.float_type)

            output = self.model(data, time, day)

            if isinstance(output, tuple):
                output = output[0]
            output = self.rescaler(output)

            error = error + self.loss(output, target)
            output, target = pt_utils.mask_target(output, target)
            if train:
                criterion = self.criterion(output, target)
                self.optimizer.zero_grad()
                criterion.backward()
                clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
            del output
        return error

    def run(self, data_train, data_eval):
        for epoch in range(self.epoches):
            error_train = self.run_epoch(data_train, train=True)
            error_eval = self.run_epoch(data_eval, train=False)
            print(f'Epoch: {epoch}',
                  f'train: {error_train}',
                  f'valid: {error_eval}',
                  sep='\n')
