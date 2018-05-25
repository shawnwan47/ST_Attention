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
        self.teach_annealing = 0.01 ** (1 / epoches)

    def eval(self, dataloader, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        metrics = MetricList()
        # infos = []
        for data_num, data_cat, target in dataloader:
            if self.cuda:
                data_num = data_num.cuda()
                data_cat = data_cat.cuda()
                target = target.cuda()
            teach = self.teach if train else 0
            output = self.model(data_num, data_cat, teach=teach)
            if isinstance(output, tuple):
                output = output[0]
            output = self.rescaler(output)

            metrics += self.metrics(output, target)
            # train
            if train:
                output, target = mask_target(output, target)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            del output
        # if infos:
        #     infos = [torch.cat(info) for info in zip(*infos)]
        return metrics

    def run(self, data_train, data_valid, data_test):
        for epoch in range(self.epoches):
            error_train = self.eval(data_train, train=True)
            error_valid = self.eval(data_valid)
            error_test = self.eval(data_test)
            print(f'Epoch: {epoch}',
                  f'train: {error_train}',
                  f'valid: {error_valid}',
                  f'test:  {error_test}',
                  f'teach ratio: {self.teach}',
                  f'learning rate: {self.optimizer.param_groups[0]["lr"]}',
                  sep='\n')
            self.scheduler.step()
            self.teach *= self.teach_annealing
