from constants import MODEL_PATH

def add_data(args):
    # data attribute
    args.add_argument('-dataset', type=str, default='BJ_highway',
                      choices=['LA_highway', 'BJ_highway', 'BJ_metro'])

    args.add_argument('-freq', type=int, default=15)
    args.add_argument('-start', type=int, default=6)
    args.add_argument('-end', type=int, default=23)
    args.add_argument('-past', type=int, default=8)
    args.add_argument('-future', type=int, default=4)

    args.add_argument('-vertices', type=int)


def add_train(args):
    # gpu
    args.add_argument('-cuda', action='store_true')
    args.add_argument('-gpuid', type=int, default=3)
    args.add_argument('-seed', type=int, default=47)
    args.add_argument('-eps', type=float, default=1e-8)
    # optimization
    args.add_argument('-loss', type=str, default='mae',
                      choices=['mae', 'rmse'])
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
    args.add_argument('-bsz', type=int, default=16)


def add_model(args):
    # general
    args.add_argument('-framework', type=str, default='seq2seq',
                      choices=['seq2seq', 'tensor2tensor'])
    args.add_argument('-model', type=str)
    args.add_argument('-nin', type=int)
    args.add_argument('-nout', type=int)
    args.add_argument('-nlayers', type=int, default=1)
    args.add_argument('-nhid', type=int, default=64)
    args.add_argument('-pdrop', type=float, default=0.2)
    # Embedding
    args.add_argument('-day_count', type=int, default=7)
    args.add_argument('-day_size', type=int, default=4)
    args.add_argument('-time_count', type=int)
    args.add_argument('-time_size', type=int, default=16)
    # Attention
    args.add_argument('-attn_type', type=str, default='dot',
                      choices=['dot', 'global', 'mlp', 'multi'])
    args.add_argument('-head', type=int, default=4)
    # Save path
    args.add_argument('-path', type=str)


def update(args):
    # data
    args.time_count = (args.end - args.start) * 60 // args.freq

    # path
    name = args.model
    name += '_hid' + str(args.nhid)
    name += '_lay' + str(args.nlayers)
    name += '_head' + str(args.head)
    args.path = MODEL_PATH + args.dataset + '/' + name
