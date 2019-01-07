import argparse
from collections import OrderedDict
from constants import MODEL_PATH
import torch


class Config:
    def __init__(self, **kwargs):
        self._add_data()
        self._add_model()
        self._add_optim()
        self.set_kwargs(**kwargs)

    def set_kwargs(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)
        self._set_data()
        self._set_model()
        self._set_cuda()

    def _add_data(self):
        # data
        self.dataset = 'LA'
        self.bday = False
        self.start = 0
        self.end = 24
        self.history = 60
        self.horizon = 60
        self.num_nodes = None
        self.freq = None
        self.num_times = None
        self.horizons = []
        self.metrics = ['mae']

    def _add_model(self):
        # model
        self.path = MODEL_PATH
        self.model = 'Transformer'
        # general
        self.model_dim = None
        self.num_layers = 2
        self.dropout = 0.2
        # Embedding
        self.weekday_dim = 8
        self.time_dim = 16
        self.node_dim = 32
        # RNN
        self.rnn_type = 'GRU'
        # Attention
        self.head_count = 4
        self.mask = False
        # GCN
        self.hops = 3

    def _add_optim(self):
        # device
        self.cuda = False
        self.seed = 47
        # optimization
        self.criterion = 'SmoothL1Loss'
        self.lr = 0.001
        self.weight_decay = 1e-5
        # run
        self.batch_size = 64
        self.epoches = 100

    def _set_cuda(self):
        self.config.cuda &= torch.cuda.is_available()
        if self.config.cuda:
            torch.cuda.set_device(self.config.gpuid)
            print(f'Using GPU: {self.config.gpuid}')
        else:
            print('Using CPU')
        if self.config.seed is not None:
            if self.config.cuda:
                torch.cuda.manual_seed(self.config.seed)
            else:
                torch.manual_seed(self.config.seed)

    def _set_data(self):
        if self.dataset == 'BJ_metro':
            self.freq = 15
            self.num_nodes = 536
            self.metrics.append('wape')
        elif self.dataset == 'BJ_highway':
            self.freq = 15
            self.num_nodes = 264
            self.metrics.append('wape')
        elif self.dataset == 'LA':
            self.freq = 5
            self.num_nodes = 207
            self.metrics.append('mape')

        self.num_times = (self.end - self.start) * 60 // self.freq
        self.history //= self.freq
        self.horizon //= self.freq
        if self.horizon == 4:
            self.horizons = [0, 1, 2, 3]
        else:
            self.horizons = [0, 2, 5, 11]
        self.freq = str(self.freq) + 'min'

    def _set_model(self):
        # model name
        self.path += self.dataset + '/'
        self.path += 'BDay' if self.bday else ''
        self.path += 'Lay' + str(self.num_layers)
        self.path += 'Dim' + str(self.model_dim)
