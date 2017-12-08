import numpy as np
import torch
from torch.autograd import Variable

import Config
import Models
import Utils
from Consts import MODEL_PATH


config = Config.Config('Seq2Seq')
config.add_optim()
config.add_transformer()
args = config.parse_args()
if torch.cuda.is_available() and not args.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")
if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

# DATA
(flow_train, flow_valid, flow_test,
 daytime_train, daytime_valid, daytime_test,
 flow_mean, flow_std) = Utils.load_data(args)

flow_train = Variable(flow_train)
flow_valid = Variable(flow_valid, volatile=True)
flow_test = Variable(flow_test, volatile=True)
daytime_train = Variable(daytime_train)
daytime_valid = Variable(daytime_valid, volatile=True)
daytime_test = Variable(daytime_test, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)


def denormalize(flow):
    return flow * flow_std + flow_mean


# MODEL
print("Model: %s" % (Config.modelname(args)))
model = Models.Seq2Seq(args)
model = model.cuda() if args.gpuid else model

# LOSS
criterion = getattr(torch.nn, args.loss)()

# OPTIM
optimizer = getattr(torch.optim, args.optim)(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.patience, verbose=True)


# training
past, future = args.past, args.future
if args.yesterday:
    past += flow_train.size(0) // 2


def train_model(flow):
    loss_train = []
    for day in torch.randperm(flow.size(1)):
        src = flow[:past, day].unsqueeze(1)
        tgt = flow[past + 1:, day].unsqueeze(1)
        inp = flow[past:-1, day].unsqueeze(1)
        out = model(src, inp, teach=True)
        out = denormalize(out[0])
        tgt = denormalize(tgt)
        loss = criterion(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()
        loss_train.append(loss.data[0])
    return sum(loss_train) / len(loss_train)


def eval_model(flow):
    loss_eval = []
    for t in range(past, flow.size(0) - future - 1):
        src = flow[:t]
        tgt = flow[t + 1:t + future + 1]
        inp = flow[t:t + future]
        out = model(src, inp, teach=False)
        out = denormalize(out[0])
        tgt = denormalize(tgt)
        loss = criterion(out, tgt)
        loss_eval.append(loss.data[0])
    return sum(loss_eval) / len(loss_eval)


for epoch in range(args.nepoch):
    loss_train = train_model(flow_train)
    loss_valid = eval_model(flow_valid)
    loss_test = eval_model(flow_test)

    print('Epoch: %d train: %.4f valid: %.4f test: %.4f' % (
        epoch, loss_train, loss_valid, loss_test))

    scheduler.step(loss_valid)

# torch.save(model.cpu(), MODEL_PATH + Config.modelname(args))

# save test results
modelpath = MODEL_PATH + Config.rnnname(args)
src = flow_test[:past]
tgt = flow_test[past + 1:]
inp = flow_test[past:-1]
out = model(src, inp, teach=True)
np.save(modelpath + '_tgt', Utils.var2np(denormalize(tgt)))
np.save(modelpath + '_out', Utils.var2np(denormalize(out[0])))
if args.attn:
    np.save(modelpath + '_att', Utils.var2np(out[-1]))
