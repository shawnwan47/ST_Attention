from constants import EPS
from lib.Loss import MetricList
from lib.pt_utils import mask_target


class Rescaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input * (self.std + EPS)) + self.mean


class Trainer:
    def __init__(self, model, rescaler, criterion, metrics,
                 optimizer, scheduler, epoches, cuda):
        self.model = model
        self.rescaler = rescaler
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cuda = cuda
        self.epoches = epoches
        self.teach = 1

    def eval(self, dataloader, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        metrics = MetricList()
        infos = []
        for data_num, data_cat, target in dataloader:
            if self.cuda:
                data_num = data_num.cuda()
                data_cat = data_cat.cuda()
                target = target.cuda()
            teach = self.teach if train else 0
            output = self.model(data_num, data_cat, teach=teach)
            if isinstance(output, tuple):
                output, info = output[0], output[1:]
                infos.append(info)
            output = self.rescaler(output)
            metrics += self.metrics(output, target)
            # optimization
            if train:
                output, target = mask_target(output, target)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if infos:
            infos = [torch.cat(info) for info in zip(*infos)]
        return metrics, infos

    def run(self, data_train, data_valid, data_test):
        for epoch in range(self.epoches):
            self.teach = 1 - (epoch / self.epoches)
            error_train, _ = self.eval(data_train, train=True)
            error_valid, _ = self.eval(data_valid)
            error_test, infos = self.eval(data_test)
            print(f'Epoch: {epoch}',
                  f'train: {error_train}',
                  f'valid: {error_valid}',
                  f'test:  {error_test}',
                  sep='\n')
            self.scheduler.step()
