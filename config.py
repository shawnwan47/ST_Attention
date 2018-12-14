import argparse
from collections import OrderedDict
from constants import MODEL_PATH


class Args(argparse.ArgumentParser):
    def __init__(self, *args):
        super().__init__()
        self._add_data()
        self._add_model()
        self._add_optim()
        self.parse_args(*args)
        self._update_data()
        self._update_model()

    def _add_data(self):
        self.add_argument('-dataset', default='LA', choices=['LA', 'BJ_highway', 'BJ_metro'])
        self.add_argument('-num_nodes', type=int)

        self.add_argument('-start', type=int, default=6)
        self.add_argument('-end', type=int, default=22)
        self.add_argument('-freq', type=int, default=5)
        self.add_argument('-num_times', type=int)

        self.add_argument('-history', type=int, default=60)
        self.add_argument('-horizon', type=int, default=60)
        self.add_argument('-horizons', nargs='+', default=[5, 15, 30, 60])

        self.add_argument('-bday', action='store_true')

        self.add_argument('-metrics', nargs='+', default=['mae'])


    def _add_model(self):
        # framework and model
        self.add_argument('-model')
        self.add_argument('-encoder')
        self.add_argument('-decoder')
        self.add_argument('-paradigm', choices=['spatial', 'temporal', 'st'])
        self.add_argument('-framework', choices=['seq2seq', 'seq2vec', 'vec2vec'])

        # general
        self.add_argument('-hidden_size', type=int)
        self.add_argument('-output_size', type=int)
        self.add_argument('-num_layers', type=int, default=2)
        self.add_argument('-dropout', type=float, default=0.2)

        # Embedding
        self.add_argument('-embedding_dim', type=int)
        # RNN
        self.add_argument('-rnn_type', default='GRU',
                          choices=['RNN', 'GRU', 'LSTM'])
        # Attention
        self.add_argument('-attn_type', default='dot',
                          choices=['dot', 'general', 'mlp'])
        self.add_argument('-head_count', type=int, default=4)
        self.add_argument('-mask', action='store_true')
        # DCRNN
        self.add_argument('-hops', type=int, default=3)
        # Save path
        self.add_argument('-path')


    def _add_optim(self):
        # device
        self.add_argument('-cuda', action='store_true')
        self.add_argument('-gpuid', type=int, default=3)
        self.add_argument('-seed', type=int, default=47)
        # optimization
        self.add_argument('-criterion', default='SmoothL1Loss',
                          choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])
        self.add_argument('-optim', default='Adam', choices=['SGD', 'Adam'])
        self.add_argument('-lr', type=float, default=0.001)
        self.add_argument('-min_lr', type=float, default=1e-6)
        self.add_argument('-weight_decay', type=float, default=1e-5)

        # run
        self.add_argument('-batch_size', type=int, default=64)
        self.add_argument('-epoches', type=int, default=100)
        self.add_argument('-iterations', type=int, default=200)


    def _set_args(self, key, value):
        if getattr(self, key) is None:
            setattr(self, key, value)


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
