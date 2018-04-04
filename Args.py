def add_data(args):
    # data attribute
    args.add_argument('-dataset', type=str, default='highway',
                      choices=['highway', 'metro'])
    args.add_argument('-inp', type=str, default='OD')
    args.add_argument('-out', type=str, default='OD')

    args.add_argument('-freq', type=int, default=15)
    args.add_argument('-start', type=int, default=360)
    args.add_argument('-past', type=int, default=120)
    args.add_argument('-future', type=int, default=60)

    args.add_argument('-num_day', type=int, default=7)
    args.add_argument('-num_time', type=int)
    args.add_argument('-num_loc', type=int)

def add_train(args):
    # gpu
    args.add_argument('-gpuid', type=int, default=0)
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)
    # optimization
    args.add_argument('-crit', type=str, default='MSELoss',
                      choices=['MSEloss', 'L1Loss'])
    args.add_argument('-optim', type=str, default='Adam',
                      choices=['SGD', 'Adam'])
    args.add_argument('-lr', type=float, default=0.001)
    args.add_argument('-min_lr', type=float, default=1e-6)
    args.add_argument('-weight_decay', type=float, default=1e-5)
    args.add_argument('-max_grad_norm', type=float, default=1)

    # run
    args.add_argument('-test', action='store_true')
    args.add_argument('-epoches', type=int, default=100)
    args.add_argument('-iterations', type=int, default=1)
    args.add_argument('-batch_size', type=int, default=256)
    args.add_argument('-print_epoches', type=int, default=1)


def add_model(args):
    # general
    args.add_argument('-model', type=str, default='Isolation',
                      choices=['Isolation', 'Transformer', 'AttentionFusion'])
    args.add_argument('-num_layers', type=int, default=1)
    args.add_argument('-hidden_size', type=int, default=64)
    args.add_argument('-dropout', type=float, default=0.2)
    # Embedding
    args.add_argument('-emb_size', type=int, default=64)
    # Attention
    args.add_argument('-att_type', type=str, default='dot',
                      choices=['dot', 'add', 'general', 'mlp'])
    args.add_argument('-head', type=int, default=4)
    args.add_argument('-map_type', type=str, default='lin',
                      choices=['lin', 'mlp', 'res'])
    args.add_argument('-res', action='store_false')
    args.add_argument('-mlp', action='store_false')
    # model name to be updated
    args.add_argument('-path', type=str)


def update(args):
    # data
    args.num_time = (1440 - args.future - args.start) // args.freq
    args.flow_size_in = args.past // args.freq
    args.flow_size_out = args.future // args.freq
    if args.dataset is 'highway':
        args.num_loc = 264
    elif args.dataset is 'metro':
        args.num_loc = 536

    # path
    path = args.model
    # Embedding
    path += 'Emb' + str(args.emb_size)
    # Att
    path += 'Map' + args.map_type
    path += 'Att' + args.att_type
    path += 'Head' + str(args.head)
    path += 'Res' if args.res else ''
    # General
    path += 'Lay' + str(args.num_layers)
    path += 'Hin' + str(args.hidden_size)
    path += 'Past' + str(args.past)
    path += 'Future' + str(args.future)
    args.path = path
