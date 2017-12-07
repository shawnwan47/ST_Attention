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
        self.add_argument('-start_time', type=int, default=5)
        self.add_argument('-end_time', type=int, default=23)
        self.add_argument('-gran', type=int, default=15)
        self.add_argument('-past', type=int, default=4)
        self.add_argument('-future', type=int, default=4)

    def add_rnn(self):
        self.add_argument('-rnn_type', type=str, default='GRU',
                          choices=['GRU', 'LSTM', 'RNN'])
        self.add_argument('-ndim', type=int, default=538)
        self.add_argument('-nhid', type=int, default=1024)
        self.add_argument('-nlay', type=int, default=1)
        self.add_argument('-pdrop', type=float, default=0.2)
        self.add_argument('-bidirectional', action='store_true')

    def add_attention(self):
        self.add_argument('-attention', action='store_true')
        self.add_argument('-attention_type', type=str, default='mlp',
                          choices=['dot', 'general', 'mlp'])
        self.add_argument('-context_length', type=int, default=0)

    def add_optim(self):
        self.add_argument('-optim_method', type=str, default='sgd',
                          choices=['sgd', 'adagrad', 'adadelta', 'adam'])
        self.add_argument('-lr', type=float, default=0.1)
        self.add_argument('-lr_min', type=float, default=1e-8)
        self.add_argument('-lr_decay', type=float, default=0.1)
        self.add_argument('-patience', type=int, default=5)
        self.add_argument('-weight_decay', type=float, default=1e-5)
        self.add_argument('-max_grad_norm', type=float, default=1.)
        self.add_argument('-beta1', type=float, default=0.9)
        self.add_argument('-beta2', type=float, default=0.98)

    def add_train(self):
        self.add_argument('-nepoch', type=int, default=100)
        self.add_argument('-niter', type=int, default=10)

    def add_loss(self):
<<<<<<< HEAD
        self.add_argument('-loss', type=str, default='MSE',
                          choices=['MSE', 'WAPE', 'MAPE'])

    def add_plot(self):
        self.add_argument('-nstation', type=int, default=4)
        self.add_argument('-istation', type=int, default=0)
=======
        self.add_argument('-loss', type=str, default='WAPE',
                          choices=['MSELoss', 'WAPE', 'MAPE'])

    def add_plot(self):
        self.add_argument('-nstation', type=int, default=4)
>>>>>>> 4040e05dbfdeb87d79b41c8070ec3291c5e46673
