from torch.nn.utils import clip_grad_norm
from utils.loss import Error


class Trainer:
    def __init__(self, model, loss, optimizer, scheduler, max_grad_norm, cuda):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.cuda = cuda

    def train(self, dataloader):
        self.model.train()
        error_total = utils.loss.Error()
        for _ in range(self.iterations):
            for data_num, data_cat, target in dataloader:
                if self.cuda:
                    data_num = data_num.cuda()
                    data_cat = data_cat.cuda()
                    target = target.cuda()
                output = model(data_num, data_cat)
                if isinstance(output, tuple):
                    output = output[0]
                error = self.loss(output, target)
                error_total.update(error)
                loss = getattr(error, self.loss.loss)
                # optimization
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
        return error_total


    def eval(self, dataloader):
        self.model.eval()
        error_total = Error()
        infos = []
        for data_num, data_cat, target in dataloader:
            output = self.model(data_num, data_cat)
            if isinstance(output, tuple):
                output, more = output[0], output[1:]
                infos.append(more)
            error = loss(output, target)
            error_total.update(error)
        if infos:
            infos = [torch.cat(info, 0).cpu().data for info in zip(*infos)]
        return error_total, infos

    def run_epoch(self, epoch, data_train, data_valid, data_test):
        error_train = self.train(data_train)
        error_valid, _ = self.eval(data_valid)
        error_test, infos = self.eval(data_test)
        print(epoch, error_train, error_valid, err_test, sep='\t')
        self.scheduler.step(loss_valid)
