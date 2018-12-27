import argparse
from collections import OrderedDict
from constants import MODEL_PATH
import torch


class Config:
    def __init__(self, *args):
        self._add_data()
        self._add_model()
        self._add_optim()
        self.parse_args(*args)
        self._update_cuda()
        self._update_data()
        self._update_model()

    def _add_data(self):
        self.dataset = 'LA'
        self.num_nodes = None
        self.start = 6
        self.end = 22
        self.freq = 5
        self.num_times = None
        self.history = 60
        self.horizon = 60
        self.horizons = [0, 15, 30, 60]
        self.bday = False


    def _add_model(self):
        # framework and model
        self.model = None
        self.framework = None

        # general
        self.hidden_size = None
        self.output_size = None
        self.num_layers = 2
        self.dropout = 0.2)

        # Embedding
        self.embedding_dim =
        self.rnn_type = RNN', 'GRU', 'LSTM'])
        # Attention
        self.attn_type = dot', 'general', 'mlp'])
        self.head_count = 4)
        self.mask =
        self.hops = 3)
        # Save path
        self.path = ):
        # device
        self.cuda = gpuid = 3)
        self.seed = 47)
        # optimization
        self.criterion = L1Loss', 'MSELoss', 'SmoothL1Loss'])
        self.optim = SGD', 'Adam'])
        self.lr = 0.001)
        self.min_lr = 1e-6)
        self.weight_decay = 1e-5)

        # run
        self.batch_size = 64)
        self.epoches = 100)
        self.iterations = 200)


    def _set_args(self, key, value):
        if getattr(self, key) is None:
            setattr(self, key, value)


    def _update_cuda(self):
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


    def _update_data(self):
        self.num_times = 1440 // self.freq
        if self.dataset == 'BJ_metro':
            self.num_nodes = 536
            self.metrics.append('wape')
        elif self.dataset == 'BJ_highway':
            self.num_nodes = 264
            self.metrics.append('wape')
        elif self.dataset == 'LA':
            self.num_nodes = 207
            self.metrics.append('mape')

        self.history //= self.freq
        self.horizon //= self.freq
        horizons = [max(0, t // self.freq - 1) for t in self.horizons]
        self.horizons = list(OrderedDict.fromkeys(horizons))
        self.freq = str(self.freq) + 'min'

        if self.bday:
            self.del_day = True


    def _update_model(self):
        assert self.encoder is not None
        self._set_args('encoder', self.model)
        self._set_args('decoder', self.encoder)

        # paradigm
        if self.model in ['RNN', 'TemporalAttention']:
            self.paradigm = 'temporal'
        elif self.model in ['MLP', 'SpatialAttention']:
            self.paradigm = 'spatial'
        else:
            self.paradigm = 'st'

        # framework
        if self.encoder in ['MLP', 'SpatialAttention']:
            self.framework = 'vec2vec'
        else:
            if self.decoder in ['MLP', 'SpatialAttention', 'DiffusionConvolution']:
                self.framework = 'seq2vec'
            else:
                self.framework = 'seq2seq'

        # hidden_size
        if self.paradigm == 'temporal':
            self.output_size = self.num_nodes
            self._set_args('hidden_size', 256)
            self._set_args('embedding_dim', 64)
        elif self.paradigm == 'spatial':
            self.output_size = self.horizon
            self._set_args('hidden_size', 64)
            self._set_args('embedding_dim', 32)
        elif self.paradigm == 'st':
            self.output_size = 1
            self._set_args('hidden_size', 32)
            self._set_args('embedding_dim', 16)

        if self.framework == 'seq2vec':
            self.output_size *= self.horizon
            self.hiddon_size *= 2

        # embedding dim
        self.time_dim = self.embedding_dim
        self.day_dim = self.embedding_dim
        self.node_dim = self.embedding_dim

        # model name
        name = self.encoder + '2' + self.decoder
        name += 'BDay' if self.bday else ''
        name += 'Lay' + str(self.num_layers)
        name += 'Hid' + str(self.hidden_size)
        name += 'Emb' + str(self.embedding_dim)
        self.path = MODEL_PATH + self.dataset + '/' + name
