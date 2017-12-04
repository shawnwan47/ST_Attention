import numpy as np

import torch
from torch.autograd import Variable

from Config import Config
import Loss
import Utils


# CONFIG
config = Config('RNN_test')
config.add_rnn()
config.add_attention()
args = config.parse_args()

# CUDA
if torch.cuda.is_available() and not args.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")
if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

# DATA
(_, _, inputs_test,
 _, _, targets_test,
 _, _, daytimes_test,
 flow_mean, flow_std) = Utils.load_data(args)
inputs_test = Variable(inputs_test, volatile=True)
targets_test = Variable(targets_test, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)


def denormalize(flow):
    return flow * flow_std + flow_mean


# MODEL
model = torch.load(Utils.modelpath(args))
model = model.cuda() if args.gpuid else model

# prediction and loss
WAPE = Loss.WAPE
MAPE = Loss.MAPE
wape = []
mape = []
outputs = []
for t in range(args.past, inputs_test.size(0) - args.future):
    src = inputs_test[:t]
    tgt = targets_test[t:t + args.future]
    inp = inputs_test[t:t + args.future]
    out = model(src, inp, teach=False)
    out = denormalize(out[0])
    tgt = denormalize(tgt)

    outputs.append(out)
    wape.append(WAPE(out, tgt).data[0])
    mape.append(MAPE(out, tgt).data[0])
wape = sum(wape) / len(wape)
mape = sum(mape) / len(mape)
print('WAPE: %.4f MAPE: %.4f' % (wape, mape))

np.save(Utils.modelpath(args) + '_tgt', Utils.var2np(targets_test[args.past:]))
np.save(Utils.modelpath(args) + '_out', Utils.var2np(torch.stack(outputs)))

# Attentions
if args.attention:
    src = inputs_test[:args.past]
    tgt = targets_test[args.past:]
    inp = inputs_test[args.past:]
    out = model(src, inp, teach=True)
    np.save(Utils.modelpath(args) + '_att', Utils.var2np(out[-1]))
