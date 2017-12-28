def add_gpu(args):
    args.add_argument('-gpuid', type=int, default=[], nargs='+')
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)


def add_data(args):
    args.add_argument('-data_type', type=str, default='highway',
                      choices=['highway', 'metro'])
    args.add_argument('-past_days', type=int, default=1)
    args.add_argument('-future', type=int, default=4)
    args.add_argument('-adj', action='store_true')
    # for metro only
    args.add_argument('-resolution', type=int, default=15)
    args.add_argument('-start_time', type=int, default=6)
    args.add_argument('-end_time', type=int, default=22)
    # args to be inferred
    args.add_argument('-past', type=int)
    args.add_argument('-daily_times', type=int)
    args.add_argument('-input_length', type=int)
    args.add_argument('-dim', type=int)
    args.add_argument('-days', type=int)
    args.add_argument('-days_train', type=int)
    args.add_argument('-days_test', type=int)


def add_loss(args):
    args.add_argument('-loss', type=str, default='L1Loss',
                      choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])


def add_optim(args):
    args.add_argument('-optim', type=str, default='SGD',
                      choices=['SGD', 'Adam', 'Adadelta', 'Adamax'])
    args.add_argument('-lr', type=float, default=0.1)
    args.add_argument('-patience', type=int, default=10)
    args.add_argument('-lr_min', type=float, default=1e-5)
    args.add_argument('-weight_decay', type=float, default=5e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)


def add_run(args):
    args.add_argument('-retrain', action='store_true')
    args.add_argument('-test', action='store_true')
    args.add_argument('-epoches', type=int, default=1000)
    args.add_argument('-batch', type=int, default=10)
    args.add_argument('-print_epoches', type=int, default=10)


def add_model(args):
    args.add_argument('-model', type=str, default='HeadAttn')
    args.add_argument('-submodel', type=str, default='Linear')
    args.add_argument('-count_submodel', type=int, default=4)
    # general
    args.add_argument('-input_size', type=int)
    args.add_argument('-output_size', type=int)
    args.add_argument('-hidden_size', type=int, default=256)
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-dropout', type=float, default=0.1)
    # regularization
    args.add_argument('-reg', action='store_true')
    args.add_argument('-reg_weight', type=float, default=5e-5)
    # Day Time
    args.add_argument('-daytime', action='store_true')
    args.add_argument('-day_size', type=int, default=16)
    args.add_argument('-time_size', type=int, default=64)
    # RNN
    args.add_argument('-rnn_type', type=str, default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    # Attention
    args.add_argument('-attn_type', type=str, default='add',
                      choices=['add', 'mul', 'mlp'])
    args.add_argument('-head', type=int, default=1)
    args.add_argument('-merge_type', type=str, default='add',
                      choices=['add', 'cat'])
    args.add_argument('-dilated', action='store_true')
    args.add_argument('-dilation', type=int, default=[], nargs='+')


def _dataset(args):
    if args.data_type == 'highway':
        args.dim = 284
        args.days = 184
        args.days_train = 120
        args.days_test = 30
        args.start_time = 0
        args.end_time = 24
        args.resolution = 15
    elif args.data_type == 'metro':
        args.dim = 536
        args.days = 22
        args.days_train = 14
        args.days_test = 4
    else:
        raise KeyError
    args.daily_times = (args.end_time - args.start_time) * 60
    args.daily_times //= args.resolution
    assert args.past_days > 0
    if args.dilated and args.num_layers == 3:
        args.past_days = 7
    args.past = args.past_days * args.daily_times
    args.input_length = args.daily_times + args.past


def _model(args):
    if args.model == 'Attention':
        args.attn = False
    # dilations for up to 4 layers
    args.dilation = [1, 8, args.daily_times, args.daily_times * 7]
    args.input_size = args.dim
    args.output_size = args.dim * args.future
    if args.daytime:
        args.input_size += args.day_size + args.time_size


def update_args(args):
    _dataset(args)
    _model(args)


def modelname(args):
    # MODEL
    path = args.model
    if 'RNN' in args.model:
        path += args.rnn_type
    if args.model == 'RNNAttn':
        path += args.attn_type
    # Attn
    if args.model == 'HeadAttn':
        path += 'Head' + str(args.head)
    if args.model == 'TemporalAttn':
        path += args.attn_type
        path += 'Dilated' if args.dilated else ''
        path += 'Head' + str(args.head)
    # General
    path += 'Lay' + str(args.num_layers)
    # path += 'Hid' + str(args.hidden_size)
    # Data
    if args.adj:
        path += 'adj'
    if args.daytime:
        path += 'Daytime'
    path += 'Future' + str(args.future)
    return path
