def add_data(args):
    # data attribute
    args.add_argument('-dataset', type=str, default='BJ_highway',
                      choices=['LA_highway', 'BJ_highway', 'BJ_metro'])
    args.add_argument('-input_od', type=str, default='OD')
    args.add_argument('-output_od', type=str, default='OD')

    args.add_argument('-freq', type=int, default=15)
    args.add_argument('-start', type=int, default=6)
    args.add_argument('-end', type=int, default=23)


def add_train(args):
    # gpu
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=0)
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)
    # optimization
    args.add_argument('-loss', type=str, default='mse',
                      choices=['mse', 'wape'])
    args.add_argument('-optim', type=str, default='Adam',
                      choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-min_lr', type=float, default=1e-6)
    args.add_argument('-weight_decay', type=float, default=1e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)

    # run
    args.add_argument('-test', action='store_true')
    args.add_argument('-retrain', action='store_true')
    args.add_argument('-epoches', type=int, default=100)
    args.add_argument('-iterations', type=int, default=1)
    args.add_argument('-batch_size', type=int, default=16)
    args.add_argument('-print_epoches', type=int, default=1)


def add_model(args):
    # general
    args.add_argument('-model', type=str)
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-hidden_size', type=int, default=64)
    args.add_argument('-dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-num_day', type=int, default=7)
    args.add_argument('-num_time', type=int)
    args.add_argument('-day_embed_size', type=int, default=4)
    args.add_argument('-time_embed_size', type=int, default=16)
    # Attention
    args.add_argument('-head', type=int, default=4)
    # Save path
    args.add_argument('-path', type=str)


def update(args):
    # data
    args.num_time = (args.end - args.start) * 60 // args.freq

    # path
    name = args.model
    name += '_hid' + str(args.hidden_size)
    name += '_lay' + str(args.num_layers)
    name += '_head' + str(args.head)
    name += '_mask' if args.mask else ''
    args.path = args.dataset + '/' + name
