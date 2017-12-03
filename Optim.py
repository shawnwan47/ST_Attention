import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(
                self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(
                self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(
                self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, weight_decay=self.weight_decay,
                betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, args):
        self.method = args.optim_method
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.lr_min = args.lr_min
        self.patience = args.patience
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.betas = [args.beta1, args.beta2]

        self.best_ppl = None
        self._stop = 0

    def _setRate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, ppl):
        if self.best_ppl is not None and ppl > self.best_ppl:
            self._stop += 1
            if self._stop > self.patience:
                self.lr = max(self.lr * self.lr_decay, self.lr_min)
                self._setRate(self.lr)
                self._stop = 0
                print("Decaying learning rate to %g" % self.lr)
        else:
            self._stop = 0
            self.best_ppl = ppl
