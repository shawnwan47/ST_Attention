def add_gpu(args):
    args.add_argument('-gpuid', default=[], nargs='+', type=int)
    args.add_argument('-seed', type=int, default=47)


def add_data(args):
    args.add_argument('-data_type', type=str, default='seq')
    args.add_argument('-gran', type=int, default=15)
    args.add_argument('-start_time', type=int, default=5)
    args.add_argument('-end_time', type=int, default=23)
    args.add_argument('-past', type=int, default=4)
    args.add_argument('-future', type=int, default=8)
    args.add_argument('-yesterday', action='store_true')


def add_loss(args):
    args.add_argument('-loss', type=str, default='L1Loss',
                      choices=['L1Loss', 'MSELoss', 'SmoothL1Loss'])


def add_optim(args):
    args.add_argument('-test', action='store_true')
    args.add_argument('-nepoch', type=int, default=100)
    args.add_argument('-optim', type=str, default='SGD',
                      choices=['SGD', 'Adam', 'Adadelta', 'Adamax'])
    args.add_argument('-lr', type=float, default=1)
    args.add_argument('-patience', type=int, default=5)
    args.add_argument('-weight_decay', type=float, default=1e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)


def add_model(args):
    args.add_argument('-model', type=str, default='RNN',
                      choices=['RNN', 'Transformer'])
    # general
    args.add_argument('-input_size', type=int, default=536)
    args.add_argument('-output_size', type=int, default=536)
    args.add_argument('-hidden_size', type=int, default=512)
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-dropout', type=float, default=0.1)
    # RNN
    args.add_argument('-rnn_type', type=str, default='RNN',
                      choices=['RNN', 'GRU', 'LSTM'])
    args.add_argument('-attn', action='store_true')
    args.add_argument('-attn_type', type=str, default='dot',
                      choices=['dot', 'general', 'mlp'])
    # Transformer
    args.add_argument('-head', type=int, default=1)
    args.add_argument('-mask_src', action='store_true')
    # Day Time size, add up to 1024
    args.add_argument('-daytime', action='store_true')
    args.add_argument('-day_size', type=int, default=16)
    args.add_argument('-time_size', type=int, default=64)


def add_plot(args):
    args.add_argument('-nstation', type=int, default=4)
    args.add_argument('-istation', type=int, default=0)


def modelname(args):
    # MODEL
    path = args.model
    # RNN
    path += args.rnn_type if args.model == 'RNN' else ''
    path += ('Attn' + args.attn_type) if (
        args.model == 'RNN' and args.attn) else ''
    # Transformer
    path += ('Head' + str(args.head)) if args.model == 'Transformer' else ''
    path += 'Masked' if args.model == 'Transformer' and args.mask_src else ''
    # general
    path += 'Layer' + str(args.num_layers)
    path += 'Hidden' + str(args.hidden_size)
    # Data
    path += 'DayTime' if args.daytime else ''
    path += 'Yesterday' if args.yesterday else ''
    return path
