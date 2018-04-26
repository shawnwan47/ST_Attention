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
        error_avg = None
        for _ in range(self.iterations):
            for data_num, data_cat, target in dataloader:
                if self.cuda:
                    data_num = data_num.cuda()
                    data_cat = data_cat.cuda()
                    target = target.cuda()
                output = model(data_num, data_cat)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.loss(output, target)
                # optimization
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
        return mse_avg / iters, wape_avg / iters


    def eval(self, dataloader):
        self.model.eval()
        mse_avg = wape_avg = iters = 0
        infos = []
        for data_num, data_cat, target in dataloader:
            output = self.model(data_num, data_cat)
            if type(output) is tuple:
                output, more = output[0], output[1:]
                infos.append(more)
            mse, wape = loss(output, target)
            mse_avg += mse.data[0]
            wape_avg += wape.data[0]
            iters += 1
        if infos:
            infos = [torch.cat(info, 0).cpu().data.numpy() for info in zip(*infos)]
        return mse_avg / iters, wape_avg / iters, infos

    def run(self, data_train, data_valid, data_test):
