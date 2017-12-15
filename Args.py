def add_gpu(args):
    args.add_argument('-gpuid', type=int, default=[], nargs='+')
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)


def add_data(args):
    args.add_argument('-data_type', type=str, default='highway',
                      choices=['highway', 'metro'])
    args.add_argument('-past_days', type=int, default=1)
    args.add_argument('-future', type=int, default=8)
    # for metro only
    args.add_argument('-gran', type=int, default=15)
    args.add_argument('-start_time', type=int, default=6)
    args.add_argument('-end_time', type=int, default=22)
    # args to be inferred
    args.add_argument('-past', type=int)
    args.add_argument('-daily_times', type=int)
    args.add_argument('-dim', type=int)
    args.add_argument('-days', type=int)
    args.add_argument('-days_train', type=int)
    args.add_argument('-days_test', type=int)


def add_loss(args):
    args.add_argument('-loss', type=str, default='L1Loss',
                      choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])


def add_optim(args):
    args.add_argument('-test', action='store_true')
    args.add_argument('-epoches', type=int, default=100)
    args.add_argument('-optim', type=str, default='SGD',
                      choices=['SGD', 'Adam', 'Adadelta', 'Adamax'])
    args.add_argument('-lr', type=float, default=1)
    args.add_argument('-patience', type=int, default=5)
    args.add_argument('-weight_decay', type=float, default=1e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)


def add_model(args):
    args.add_argument('-model', type=str, default='Attention',
                      choices=['RNN', 'Attention', 'DilatedAttention'])
    # general
    args.add_argument('-input_size', type=int)
    args.add_argument('-output_size', type=int)
    args.add_argument('-hidden_size', type=int, default=1024)
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-dropout', type=float, default=0.1)
    # Day Time size
    args.add_argument('-daytime', action='store_true')
    args.add_argument('-day_size', type=int, default=16)
    args.add_argument('-time_size', type=int, default=64)
    # RNN
    args.add_argument('-rnn_type', type=str, default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    args.add_argument('-attn', action='store_true')
    args.add_argument('-attn_type', type=str, default='dot',
                      choices=['dot', 'general', 'mlp', 'kvq'])
    # Transformer
    args.add_argument('-head', type=int, default=1)
    args.add_argument('-dilated', action='store_true')
    args.add_argument('-dilation', type=int, nargs='+', default=[])


def _dataset(args):
    if args.data_type == 'highway':
        args.dim = 286
        args.days = 184
        args.days_train = 120
        args.days_test = 30
        args.start_time = 0
        args.end_time = 24
        args.gran = 15
    elif args.data_type == 'metro':
        args.dim = 286
        args.days = 22
        args.days_train = 14
        args.days_test = 4
    else:
        raise KeyError
    args.daily_times = (args.end_time - args.start_time) * 60 // args.gran
    assert args.past_days > 0
    args.past = args.past_days * args.daily_times


def _model(args):
    if args.model == 'Attention':
        args.attn = False
        args.daytime = True
    # dilations for up to 4 layers
    args.dilation = [1, 4, 16, args.daily_times, args.daily_times * 7]
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
    # RNN
    if args.model == 'RNN':
        path += args.rnn_type
        path += 'Attn' + args.attn_type if args.attn else ''
    # Attention
    if args.model == 'Attention':
        path += ('Head' + str(args.head))
        path += 'Dilated' if args.dilated else ''
    # general
    path += 'Lay' + str(args.num_layers)
    path += 'Hid' + str(args.hidden_size)
    # Data
    if args.daytime:
        path += 'Day' + str(args.day_size) + 'Time' + str(args.time_size)
    path += 'Past' + str(args.past_days)
    return path
