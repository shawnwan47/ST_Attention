from collections import OrderedDict
from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', default='LA',
                      choices=['LA', 'BJ_highway', 'BJ_metro'])

    args.add_argument('-freq', type=int, default=5)
    args.add_argument('-start', type=int, default=6)
    args.add_argument('-end', type=int, default=22)
    args.add_argument('-bday', action='store_true')
    args.add_argument('-history', type=int, default=60)
    args.add_argument('-horizon', type=int, default=60)

    args.add_argument('-metrics', nargs='+', default=['mae'])
    args.add_argument('-horizons', nargs='+', default=[5, 15, 30, 60])

    args.add_argument('-num_days', type=int, default=7)
    args.add_argument('-num_times', type=int)
    args.add_argument('-num_nodes', type=int)

    args.add_argument('-del_time', action='store_true')
    args.add_argument('-del_day', action='store_true')
    args.add_argument('-del_node', action='store_true')


def add_model(args):
    # framework and model
    args.add_argument('-model')
    # general
    args.add_argument('-io', default='graph', choices=['graph', 'vector'])
    args.add_argument('-framework', default='seq2seq',
                      choices=['seq2seq', 'seq2vec', 'vec2vec'])
    args.add_argument('-hidden_size', type=int)
    args.add_argument('-output_size', type=int)
    args.add_argument('-num_layers', type=int, default=1)
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
    args.add_argument('-attn_type', default='dot',
                      choices=['dot', 'general', 'mlp'])
    args.add_argument('-head_count', type=int, default=4)
    args.add_argument('-mask', action='store_true')
    # DCRNN
    args.add_argument('-hops', type=int, default=3)
    # Save path
    args.add_argument('-path')


def add_train(args):
    # device
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=3)
    args.add_argument('-seed', type=int, default=47)
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
    args.add_argument('-iterations', type=int, default=100)


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

    if args.bday:
        args.del_day = True


def update_model(args):
    # io
    if args.io is 'vector':
        args.output_size = args.num_nodes
    else:
        args.output_size = 1
    if args.framework is not 'seq2seq':
        args.output_size *= args.horizon
    # model
    name = args.model
    if 'RNN' in args.model:
        name += args.rnn_type
    if args.model in ['GRARNN', 'GARNN', 'Transformer', 'STTransformer']:
        name += 'Head' + str(args.head_count)
        name += 'Masked' if args.mask else ''
    name += 'Hid' + str(args.hidden_size)
    name += 'Lay' + str(args.num_layers)
    # data
    if args.del_node:
        name += 'Node'
    if args.del_time:
        name += 'Time'
    if args.del_day:
        name += 'Day'
    name += args.freq
    args.path = MODEL_PATH + args.dataset + '/' + name
