from torch.nn.utils import clip_grad_norm_
from lib.Loss import MultiError


class Trainer:
    def __init__(self, model, loss, optimizer, scheduler, cuda):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._cuda = cuda
        self._teach = 1.
        self._epoch = 1

    def eval(self, dataloader, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        errors = MultiError()
        infos = []
        for data_num, data_cat, target in dataloader:
            if self._cuda:
                data_num = data_num.cuda()
                data_cat = data_cat.cuda()
                target = target.cuda()
            teach = self._teach if train else 0
            output = self.model(data_num, data_cat, teach=teach)
            if isinstance(output, tuple):
                output, info = output[0], output[1:]
                infos.append(info)
            loss, error = self.loss(output, target)
            errors.update(error)
            # optimization
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if infos:
            infos = [torch.cat(info) for info in zip(*infos)]
        return errors, infos

    def run_epoch(self, data_train, data_valid, data_test):
        error_train, _ = self.eval(data_train, True)
        error_valid, _ = self.eval(data_valid)
        error_test, infos = self.eval(data_test)
        print(f'Epoch: {self._epoch}')
        print(str(error_train), str(error_valid), str(error_test), sep='\n')
        self._epoch += 1
        self.scheduler.step()
        self._teach *= 0.98