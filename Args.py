def add_args(args):
    add_gpu(args)
    add_data(args)
    add_loss(args)
    add_optim(args)
    add_run(args)
    add_model(args)


def add_gpu(args):
    args.add_argument('-gpuid', type=int, default=0)
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)


def add_data(args):
    # data attribute
    args.add_argument('-data_type', type=str, default='highway',
                      choices=['highway', 'metro'])
    args.add_argument('-num_flow', type=int, default=64)
    args.add_argument('-num_day', type=int, default=7)
    args.add_argument('-num_time', type=int, default=96)
    args.add_argument('-num_loc', type=int)
    # dataset
    args.add_argument('-past', type=int, default=1)
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
    args.add_argument('-loss', type=str, default='CrossEntropyLoss',
                      choices=['L1Loss', 'CrossEntropyLoss'])


def add_optim(args):
    args.add_argument('-optim', type=str, default='Adam',
                      choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-patience', type=int, default=10)
    args.add_argument('-lr_min', type=float, default=1e-5)
    args.add_argument('-weight_decay', type=float, default=5e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)


def add_run(args):
    args.add_argument('-test', action='store_true')
    args.add_argument('-retrain', action='store_true')
    args.add_argument('-epoches', type=int, default=300)
    args.add_argument('-iterations', type=int, default=1)
    args.add_argument('-batch', type=int, default=300)
    args.add_argument('-print_epoches', type=int, default=1)


def add_model(args):
    # general
    args.add_argument('-model', type=str, default='Transformer')
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-emb_size', type=int, default=32)
    args.add_argument('-emb_merge', type=str, default='cat',
                      choices=['cat', 'sum'])
    args.add_argument('-emb_all', type=int)
    # Attention
    args.add_argument('-attn_type', type=str, default='dot',
                      choices=['dot', 'add'])
    args.add_argument('-head', type=int, default=1)


def update_args(args):
    if args.data_type == 'highway':
        args.num_loc = 284
        args.days = 184
        args.days_train = 150
        args.days_test = 15
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
    # model
    if args.emb_merge == 'cat':
        args.emb_all = 4 * args.emb_size
    else:
        args.emb_all = args.emb_size
    # run
    if args.retrain:
        args.lr /= 100


def modelname(args):
    # MODEL
    path = args.model
    # Embedding
    path += 'Emb' + str(args.emb_size) + args.emb_merge
    # Attn
    path += 'Head' + str(args.head)
    # General
    path += 'Lay' + str(args.num_layers)
    path += 'Flow' + str(args.num_flow)
    path += 'Past' + str(args.past)
    path += 'Future' + str(args.future)
    return path
