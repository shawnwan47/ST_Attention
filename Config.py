import argparse


class Config(argparse.ArgumentParser):
    def __init__(self, description=None):
        super(Config, self).__init__(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_gpu()
        self.add_data()
        self.add_loss()

    def add_gpu(self):
        self.add_argument('-gpuid', default=[], nargs='+', type=int)
        self.add_argument('-seed', type=int, default=47)

    def add_data(self):
        self.add_argument('-data_type', type=str, default='seq')
        self.add_argument('-gran', type=int, default=15)
        self.add_argument('-start_time', type=int, default=5)
        self.add_argument('-end_time', type=int, default=23)
        self.add_argument('-past', type=int, default=4)
        self.add_argument('-future', type=int, default=4)
        self.add_argument('-input_size', type=int, default=536)
        self.add_argument('-yesterday', action='store_true')

    def add_optim(self):
        self.add_argument('-nepoch', type=int, default=100)
        self.add_argument('-optim', type=str, default='SGD',
                          choices=['SGD', 'Adam', 'Adadelta', 'Adamax'])
        self.add_argument('-lr', type=float, default=0.1)
        self.add_argument('-patience', type=int, default=3)
        self.add_argument('-weight_decay', type=float, default=1e-5)
        self.add_argument('-max_grad_norm', type=float, default=1)

    def add_loss(self):
        self.add_argument('-loss', type=str, default='L1Loss',
                          choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])

    def add_rnn(self):
        self.add_argument('-rnn_type', type=str, default='GRU',
                          choices=['RNN', 'GRU', 'LSTM'])
        self.add_argument('-hidden_size', type=int, default=1024)
        self.add_argument('-num_layers', type=int, default=1)
        self.add_argument('-dropout', type=float, default=0.1)
        self.add_argument('-attn', action='store_true')
        self.add_argument('-attn_type', type=str, default='mlp',
                          choices=['dot', 'general', 'mlp'])

    def add_transformer(self):
        self.add_argument('-head_count', type=int, default=1)
        self.add_argument('-hidden_size', type=int, default=1024)
        self.add_argument('-num_layers', type=int, default=1)
        self.add_argument('-dropout', type=float, default=0.1)

    def add_plot(self):
        self.add_argument('-nstation', type=int, default=4)
        self.add_argument('-istation', type=int, default=0)


def rnnname(args):
    path = 'Yesterday' if args.yesterday else ''
    path += args.rnn_type
    path += 'Hidden' + str(args.hidden_size)
    path += 'Layer' + str(args.num_layers)
    path += ('Attn' + args.attn_type) if args.attn else ''
    return path
