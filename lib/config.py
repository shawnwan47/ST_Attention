from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', default='LA',
                      choices=['LA', 'BJ_highway', 'BJ_metro'])

    args.add_argument('-freq', type=int, default=5)
    args.add_argument('-start', type=int, default=6)
    args.add_argument('-end', type=int, default=22)
    args.add_argument('-past', type=int, default=60)
    args.add_argument('-future', type=int, default=60)
    args.add_argument('-futures', nargs='+', default=[15, 30, 60])

    args.add_argument('-nodes', type=int)

    args.add_argument('-period', choices=['continous', 'daily', 'weekly'])


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
    args.add_argument('-bsz', type=int, default=256)


def add_model(args):
    # framework and model
    args.add_argument('-model', default='RNN',
                      choices=['RNN', 'RNNAttn', 'GCRNN'])
    # general parameters
    args.add_argument('-nin', type=int)
    args.add_argument('-nout', type=int)
    args.add_argument('-nlayers', type=int, default=2)
    args.add_argument('-nhid', type=int, default=256)
    args.add_argument('-pdrop', type=float, default=0.2)
    # Embedding
    args.add_argument('-day_count', type=int, default=7)
    args.add_argument('-day_size', type=int, default=8)
    args.add_argument('-time_count', type=int)
    args.add_argument('-time_size', type=int, default=16)
    # RNN
    args.add_argument('-rnn_type', default='RNN', choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', default='dot',
                      choices=['dot', 'global', 'mlp', 'multi'])
    args.add_argument('-head', type=int, default=4)
    # Save path
    args.add_argument('-path')


def update_data(args):
    # data
    args.time_count = 1440 // args.freq
    if args.dataset == 'BJ_metro':
        args.nodes = 536
    elif args.dataset == 'BJ_highway':
        args.nodes = 264
    elif args.dataset == 'LA':
        args.nodes = 207

    args.past //= args.freq
    args.future //= args.freq
    args.futures = [t // args.freq - 1 for t in args.futures]


def update_model(args):
    embed_size = args.day_size + args.time_size
    if args.model in ['RNN', 'RNNAttn']:
        args.nin = args.nodes + embed_size
        args.nout = args.nodes
    if args.model == 'GCRNN':
        args.nin = 1
        args.nout = 1

    # path
    name = args.model
    name += '_hid' + str(args.nhid)
    name += '_lay' + str(args.nlayers)
    if args.model == 'Transformer':
        name += '_head' + str(args.head)
    args.path = MODEL_PATH + args.dataset + '/' + name
