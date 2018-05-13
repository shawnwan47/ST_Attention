from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', default='BJ_highway',
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
    args.add_argument('-epoches', type=int, default=100)
    args.add_argument('-iters', type=int, default=1)
    args.add_argument('-batch_size', type=int, default=256)


def add_model(args):
    # framework and model
    args.add_argument('-model', default='RNN',
                      choices=['RNN', 'RNNAttn', 'GCRNN'])
    # general parameters
    args.add_argument('-input_size', type=int)
    args.add_argument('-output_size', type=int)
    args.add_argument('-num_layers', type=int, default=2,
                      choices=[1, 2, 3])
    args.add_argument('-hidden_size', type=int, default=256,
                      choices=[16, 32, 64, 128, 256, 512])
    args.add_argument('-p_dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-flow_size', type=int, default=16)
    args.add_argument('-day_size', type=int, default=16)
    args.add_argument('-time_size', type=int, default=16)
    args.add_argument('-node_size', type=int, default=16)
    # RNN
    args.add_argument('-rnn_type', default='RNN', choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', default='dot',
                      choices=['dot', 'global', 'mlp', 'multi'])
    args.add_argument('-head_count', type=int, default=4)
    # Save path
    args.add_argument('-path')


def update_data(args):
    args.time_count = 1440 // args.freq
    if args.dataset == 'BJ_metro':
        args.node_count = 536
    elif args.dataset == 'BJ_highway':
        args.node_count = 268
    elif args.dataset == 'LA':
        args.node_count = 207

    args.past //= args.freq
    args.future //= args.freq
    args.futures = [t // args.freq - 1 for t in args.futures]
    args.freq = str(args.freq) + 'min'


def update_model(args):
    embed_size = args.day_size + args.time_size
    if args.model in ['RNN', 'RNNAttn']:
        args.input_size = args.node_count + embed_size
        args.output_size = args.node_count
    if args.model in ['GCRNN', 'GARNN']:
        args.input_size = sum(args.flow_size, args.day_size, args.time_size, args.node_size)
        args.output_size = 1

    # path
    name = args.model
    name += '_hid' + str(args.hidden_size)
    name += '_lay' + str(args.num_layers)
    if args.model == 'Transformer':
        name += '_head' + str(args.head)
    args.path = MODEL_PATH + args.dataset + '/' + name
