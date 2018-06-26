import math
from collections import OrderedDict
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

    args.add_argument('-metrics', nargs='+', default=['mae'])
    args.add_argument('-horizons', nargs='+', default=[5, 15, 30, 60])

    args.add_argument('-num_days', type=int, default=7)
    args.add_argument('-num_times', type=int)
    args.add_argument('-num_nodes', type=int)
    args.add_argument('-num_dists', type=int, default=64)

    args.add_argument('-discretize', action='store_true')
    args.add_argument('-use_time', action='store_true')
    args.add_argument('-use_day', action='store_true')
    args.add_argument('-use_node', action='store_true')
    args.add_argument('-od', action='store_true')


def add_device(args):
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=3)
    args.add_argument('-seed', type=int, default=47)


def add_train(args):
    # optimization
    args.add_argument('-criterion', default='SmoothL1Loss',
                      choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])
    args.add_argument('-optim', default='Adam', choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-min_lr', type=float, default=1e-6)
    args.add_argument('-weight_decay', type=float, default=1e-5)

    # run
    args.add_argument('-test', action='store_true')
    args.add_argument('-batch_size', type=int)
    args.add_argument('-epoches', type=int, default=100)


def add_model(args):
    # framework and model
    args.add_argument('-model')
    # general
    args.add_argument('-output_size', type=int)
    args.add_argument('-num_layers', type=int, default=2)
    args.add_argument('-hidden_size', type=int)
    args.add_argument('-dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-day_dim', type=int, default=16)
    args.add_argument('-time_dim', type=int, default=16)
    args.add_argument('-node_dim', type=int, default=16)
    args.add_argument('-time_dist_dim', type=int, default=16)
    args.add_argument('-node_dist_dim', type=int, default=16)
    # RNN
    args.add_argument('-rnn_type', default='GRU',
                      choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', default='general',
                      choices=['dot', 'general', 'mlp'])
    args.add_argument('-head_count', type=int, default=4)
    # DCRNN
    args.add_argument('-hops', type=int, default=3)
    # Save path
    args.add_argument('-path')


def set_args(args, key, value):
    if getattr(args, key) is None:
        setattr(args, key, value)


def update_data(args):
    args.num_times = 1440 // args.freq
    if args.dataset == 'BJ_metro':
        args.num_nodes = 536
        args.metrics.append('wape')
    elif args.dataset == 'BJ_highway':
        args.num_nodes = 264
        args.metrics.append('wape')
    elif args.dataset == 'LA':
        args.num_nodes = 207
        args.metrics.append('mape')

    args.history //= args.freq
    args.horizon //= args.freq
    horizons = [max(0, t // args.freq - 1) for t in args.horizons]
    args.horizons = list(OrderedDict.fromkeys(horizons))
    args.freq = str(args.freq) + 'min'


def update_model(args):
    if args.model in ['RNN',
                      'RNNAttn',
                      'Transformer',
                      'RelativeTransformer']:
        args.output_size = args.num_nodes
        set_args(args, 'batch_size', 64)
        set_args(args, 'hidden_size', 256)
    elif args.model in ['DCRNN',
                        'GARNN',
                        'GRARNN',
                        'STTransformer',
                        'RelativeSTTransformer']:
        args.output_size = 1
        set_args(args, 'batch_size', 16)
        set_args(args, 'hidden_size', 64)
    else:
        raise NameError('Model {model} invalid.'.format(model))

    # path
    name = args.model
    if 'RNN' in args.model:
        name += args.rnn_type
    if args.model in ['GARNN', 'Transformer']:
        name += 'Head' + str(args.head_count)
    name += 'Hid' + str(args.hidden_size)
    name += 'Lay' + str(args.num_layers)
    if args.use_node:
        name += 'Node'
    if args.use_time:
        name += 'Time'
    if args.use_day:
        name += 'Day'
    name += args.freq
    args.path = MODEL_PATH + args.dataset + '/' + name
