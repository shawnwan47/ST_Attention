from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', default='LA',
                      choices=['LA', 'BJ_highway', 'BJ_metro'])

    args.add_argument('-freq', type=int, default=5)
    args.add_argument('-start', type=int, default=0)
    args.add_argument('-end', type=int, default=24)
    args.add_argument('-past', type=int, default=60)
    args.add_argument('-future', type=int, default=60)
    args.add_argument('-futures', nargs='+', default=[15, 30, 60])

    args.add_argument('-day_count', type=int, default=7)
    args.add_argument('-time_count', type=int)
    args.add_argument('-node_count', type=int)


def add_train(args):
    # gpu
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=3)
    args.add_argument('-seed', type=int, default=47)
    # optimization
    args.add_argument('-loss', default='mae', choices=['mae', 'rmse'])
    args.add_argument('-optim', default='Adam', choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-min_lr', type=float, default=1e-6)
    args.add_argument('-weight_decay', type=float, default=1e-5)

    # run
    args.add_argument('-test', action='store_true')
    args.add_argument('-retrain', action='store_true')
    args.add_argument('-batch_size', type=int, default=128)
    args.add_argument('-epoches', type=int, default=100)
    args.add_argument('-iters', type=int, default=1)


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
    args.add_argument('-day_size', type=int, default=4)
    args.add_argument('-time_size', type=int, default=16)
    args.add_argument('-node_size', type=int, default=16)
    # RNN
    args.add_argument('-rnn_type', default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', default='general',
                      choices=['dot', 'general', 'mlp'])
    args.add_argument('-head_count', type=int, default=1)
    # DCRNN
    args.add_argument('-hops', type=int, default=3)
    args.add_argument('-uni', action='store_false')
    # Save path
    args.add_argument('-path')


def _set_args(args, kwargs):
    for key, value in kwargs.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def get_model_config(model):
    assert model in ['RNN', 'GCRNN']
    if model == 'RNN':
        return {'hidden_size': 256}
    elif model == 'GCRNN':
        return {'hidden_size': 64}


def update_data(args):
    args.time_count = 1440 // args.freq
    if args.dataset == 'BJ_metro':
        args.node_count = 536
    elif args.dataset == 'BJ_highway':
        args.node_count = 264
    elif args.dataset == 'LA':
        args.node_count = 207

    args.past //= args.freq
    args.future //= args.freq
    args.futures = [t // args.freq - 1 for t in args.futures]
    args.freq = str(args.freq) + 'min'


def update_model(args):
    if args.model in ['RNN', 'RNNAttn']:
        args.input_size = args.node_count + args.day_size + args.time_size
        args.output_size = args.node_count
        _set_args(args, get_model_config('RNN'))
    elif args.model in ['DCRNN', 'GARNN']:
        args.input_size = args.day_size + args.time_size + args.node_size + 1
        args.output_size = 1
        _set_args(args, get_model_config('GCRNN'))
    else:
        raise NameError('model {0} invalid!'.format(args.model))

    # path
    name = args.model
    name += args.freq
    name += '_hid' + str(args.hidden_size)
    name += '_lay' + str(args.num_layers)
    if args.model == 'RNN':
        name += 'rnn_' + args.rnn_type
    if args.model == 'Transformer':
        name += '_head' + str(args.head)
    args.path = MODEL_PATH + args.dataset + '/' + name
