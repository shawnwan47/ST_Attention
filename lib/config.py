import math
from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', default='LA',
                      choices=['LA', 'BJ_highway', 'BJ_metro'])

    args.add_argument('-freq', type=int, default=5)
    args.add_argument('-start', type=int, default=0)
    args.add_argument('-end', type=int, default=24)
    args.add_argument('-history', type=int, default=60)
    args.add_argument('-horizon', type=int, default=60)

    args.add_argument('-metrics', nargs='+', default=['mae', 'rmse'])
    args.add_argument('-horizons', nargs='+', default=[15, 30, 60])

    args.add_argument('-num_weekday', type=int, default=7)
    args.add_argument('-num_time', type=int)
    args.add_argument('-num_node', type=int)
    args.add_argument('-num_time_dist', type=int)
    args.add_argument('-num_node_dist', type=int)

    args.add_argument('-discretize', action='store_true')
    args.add_argument('-use_time', action='store_true')
    args.add_argument('-use_weekday', action='store_true')
    args.add_argument('-use_node', action='store_true')
    args.add_argument('-use_time_dist', action='store_true')
    args.add_argument('-use_node_dist', action='store_true')
    args.add_argument('-od', action='store_true')


def add_train(args):
    # gpu
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=3)
    args.add_argument('-seed', type=int, default=47)
    # optimization
    args.add_argument('-criterion', default='L1Loss',
                      choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])
    args.add_argument('-optim', default='Adam', choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-min_lr', type=float, default=1e-6)
    args.add_argument('-weight_decay', type=float, default=1e-5)

    # run
    args.add_argument('-test', action='store_true')
    args.add_argument('-retrain', action='store_true')
    args.add_argument('-batch_size', type=int, default=64)
    args.add_argument('-epoches', type=int, default=100)


def add_model(args):
    # framework and model
    args.add_argument('-model')
    # general
    args.add_argument('-input_size', type=int)
    args.add_argument('-output_size', type=int)
    args.add_argument('-num_layers', type=int, default=2,
                      choices=[1, 2, 3])
    args.add_argument('-hidden_size', type=int)
    args.add_argument('-dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-weekday_dim', type=int, default=16)
    args.add_argument('-time_dim', type=int, default=16)
    args.add_argument('-node_dim', type=int, default=16)
    args.add_argument('-time_dist_dim', type=int, default=16)
    args.add_argument('-node_dist_dim', type=int, default=16)
    # RNN
    args.add_argument('-rnn_type', default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', choices=['dot', 'general', 'mlp'])
    args.add_argument('-head_count', type=int)
    # DCRNN
    args.add_argument('-hops', type=int)
    args.add_argument('-uni', action='store_true')
    # Save path
    args.add_argument('-path')


def _set_args(args, kwargs):
    for key, value in kwargs.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def get_model_config(model):
    if model == 'RNN':
        config = {'hidden_size': 256}
    elif model == 'RNNAttn':
        config = {
            'hidden_size': 256,
            'attn_type': 'general'
        }
    elif model == 'DCRNN':
        config = {
            'hidden_size': 16,
            'hops': 3,
        }
    elif model == 'GARNN':
        config = {
            'hidden_size': 16,
            'head_count': 1
        }
    else:
        raise NameError('Model {model} invalid.'.format(model))
    return config


def update_data(args):
    args.num_time = 1440 // args.freq
    if args.dataset == 'BJ_metro':
        args.num_node = 536
        args.metrics.append('wape')
    elif args.dataset == 'BJ_highway':
        args.num_node = 264
        args.metrics.append('wape')
    elif args.dataset == 'LA':
        args.num_node = 207
        args.metrics.append('mape')

    args.history //= args.freq
    args.horizon //= args.freq
    args.horizons = [t // args.freq - 1 for t in args.horizons]
    args.freq = str(args.freq) + 'min'
    args.num_time_dist = args.history + args.horizon
    args.num_node_dist = args.num_node


def update_model(args):
    if args.model in ['RNN', 'RNNAttn']:
        args.input_size = args.num_node
        args.output_size = args.num_node
        if args.use_time:
            args.input_size += args.time_dim
        if args.use_weekday:
            args.input_size += args.weekday_dim
        _set_args(args, get_model_config(args.model))
    elif args.model in ['DCRNN', 'GARNN']:
        args.input_size = 1
        args.output_size = 1
        if args.use_node:
            args.input_size += args.node_dim
        if args.use_time:
            args.input_size += args.time_dim
        if args.use_weekday:
            args.input_size += args.weekday_dim
        _set_args(args, get_model_config(args.model))
    else:
        raise NameError('model {0} invalid!'.format(args.model))

    # path
    name = args.model
    name += args.freq
    if args.use_node:
        name += 'Node'
    if args.use_time:
        name += 'Time'
    if args.use_weekday:
        name += 'Weekday'
    name += 'Hid' + str(args.hidden_size)
    name += 'Lay' + str(args.num_layers)
    if 'RNN' in args.model:
        name += 'RNN' + args.rnn_type
    if args.model == 'Transformer':
        name += 'Head' + str(args.head)
    args.path = MODEL_PATH + args.dataset + '/' + name
