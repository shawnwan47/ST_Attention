from torch.nn.utils import clip_grad_norm_
from utils.Loss import Error


class Trainer:
    def __init__(self, model, loss, optimizer, scheduler,
                 iters, max_grad_norm, cuda):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._iters = iters
        self._cuda = cuda
        self._epoch = 1

    def eval(self, dataloader, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        error_total = Error()
        infos = []
        for _ in range(self._iters):
            for data_num, data_cat, target in dataloader:
                if self._cuda:
                    data_num = data_num.cuda()
                    data_cat = data_cat.cuda()
                    target = target.cuda()
                teach = 0.5 if train else 0
                output = self.model(data_num, data_cat, teach=teach)
                if isinstance(output, tuple):
                    output, info = output[0], output[1:]
                    infos.append(info)
                error = self.loss(output, target)
                error_total.update(error)
                loss = getattr(error, self.loss.loss)
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                self.optimizer.step()
        if infos:
            infos = [torch.cat(info) for info in zip(*infos)]
        return error_total, infos

    def run_epoch(self, data_train, data_valid, data_test):
        error_train, _ = self.eval(data_train, True)
        error_valid, _ = self.eval(data_valid)
        error_test, infos = self.eval(data_test)
        print(self._epoch, error_train, error_valid, error_test, sep='\t')
        self.scheduler.step(getattr(error_valid, self.loss.loss))
        self._epoch += 1
