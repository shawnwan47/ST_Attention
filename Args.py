def add_gpu(args):
    args.add_argument('-gpuid', type=int, default=0)
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)


def add_data(args):
    # data attribute
    args.add_argument('-data_type', type=str, default='highway',
                      choices=['highway', 'metro'])
    args.add_argument('-num_flow', type=int, default=128)
    args.add_argument('-num_day', type=int, default=7)
    args.add_argument('-num_time', type=int)
    args.add_argument('-num_loc', type=int)
    # dataset
    args.add_argument('-past', type=int, default=4)
    args.add_argument('-future', type=int, default=1)
    args.add_argument('-past_day', action='store_true')
    args.add_argument('-max_len', type=int)
    # for metro only
    args.add_argument('-resolution', type=int, default=15)
    args.add_argument('-start_time', type=int, default=6)
    args.add_argument('-end_time', type=int, default=22)
    # args to be inferred
    args.add_argument('-days', type=int)
    args.add_argument('-days_train', type=int)
    args.add_argument('-days_test', type=int)


def add_loss(args):
    args.add_argument('-loss', type=str, default='NLLLoss2d',
                      choices=['L1Loss', 'NLLLoss2d'])


def add_optim(args):
    args.add_argument('-optim', type=str, default='SGD',
                      choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=1)
    args.add_argument('-patience', type=int, default=10)
    args.add_argument('-lr_min', type=float, default=1e-5)
    args.add_argument('-weight_decay', type=float, default=5e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)


def add_run(args):
    args.add_argument('-test', action='store_true')
    args.add_argument('-epoches', type=int, default=1000)
    args.add_argument('-batch', type=int, default=100)
    args.add_argument('-print_epoches', type=int, default=1)


def add_model(args):
    # general
    args.add_argument('-model', type=str)
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-dropout', type=float, default=0.1)
    # regularization
    args.add_argument('-reg', action='store_true')
    args.add_argument('-reg_weight', type=float, default=0.1)
    # Embedding
    args.add_argument('-emb_flow', type=int, default=32)
    args.add_argument('-emb_day', type=int, default=32)
    args.add_argument('-emb_time', type=int, default=32)
    args.add_argument('-emb_loc', type=int, default=32)
    args.add_argument('-emb_size', type=int)
    # RNN
    args.add_argument('-rnn_type', type=str, default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', type=str, default='add',
                      choices=['add', 'dot', 'mul', 'mlp'])
    args.add_argument('-head', type=int, default=1)
    args.add_argument('-merge_type', type=str, default='add',
                      choices=['add', 'cat'])


def _dataset(args):
    if args.data_type == 'highway':
        args.num_loc = 284
        args.days = 184
        args.days_train = 120
        args.days_test = 30
        args.start_time = 0
        args.end_time = 24
        args.resolution = 15
    elif args.data_type == 'metro':
        args.num_loc = 536
        args.days = 22
        args.days_train = 14
        args.days_test = 4
    else:
        raise KeyError
    args.num_time = (args.end_time - args.start_time) * 60 // args.resolution
    if args.past_day:
        args.past = args.num_time
    args.max_len = 2 * args.num_time


def _model(args):
    args.emb_size = args.emb_flow + args.emb_day + args.emb_time + args.emb_loc


def update_args(args):
    _dataset(args)
    _model(args)


def modelname(args):
    # MODEL
    path = args.model
    if args.model.startswith('En'):
        path += str(args.subnum) + args.submodel
    if 'RNN' in args.model:
        path += args.rnn_type
    if args.model == 'RNNAttn':
        path += args.attn_type
    if args.reg:
        path += 'Reg' + str(args.reg_weight)
    # Attn
    if 'Attn' in args.model:
        path += 'Head' + str(args.head)
        path += args.attn_type
    # General
    path += 'Lay' + str(args.num_layers)
    # Data
    path += 'Future' + str(args.future)
    return path
