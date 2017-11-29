import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.augograd import Variable

import Utils
import Models
from Trainer import seq2seq_attn

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--granularity', type=int, default=15)
parser.add_argument('--past', type=int, default=40)
parser.add_argument('--future', type=int, default=8)
# model
parser.add_argument('--nhid', type=int, default=512)
parser.add_argument('--nlay', type=int, default=2)
parser.add_argument('--attn_type', type=str, default='general',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('--loss', type=str, default='MAPE',
                    choices=['WAPE', 'MAPE'])
# train
parser.add_argument('--bsz', type=int, default=100)
parser.add_argument('--niter', type=int, default=10000)
parser.add_argument('--nepoch', type=int, default=10)
parser.add_argument('--iprint', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_min', type=float, default=0.00001)
parser.add_argument('--lr_decay', type=int, default=5)
# gpu
parser.add_argument('--gpuid', type=list, default=[3])
parser.add_argument('--seed', type=int, default=47)
# file
parser.add_argument('--savepath', type=str, default='save.pt')

args = parser.parse_args()

if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

# data
inputs, outputs, days, times, flow_mean, flow_std = Utils.load_flow(
    granularity=args.granularity, past=args.past, future=args.future)
inputs_train, inputs_valid, inputs_test = Utils.split_dataset(inputs)
outputs_train, outputs_valid, outputs_test = Utils.split_dataset(outputs)
inputs_train, outputs_train = Variable(inputs_train), Variable(outputs_train)
inputs_valid, outputs_valid = Variable(inputs_valid), Variable(outputs_valid)

# model
ndim = inputs_train.shape[-1]
encoder = Models.EncoderRNN(ndim, args.nhid)
decoder = Models.AttnDecoderRNN(ndim, args.nhid)

if args.gpuid:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


# training
def trainIters(encoder, decoder, bsz, niter, iprint, lr, lr_min, lr_decay):

    def get_batch(inputs, outputs, bsz):
        idx = np.random.randint(0, inputs.shape[0], bsz)
        var_inputs = np2torch(inputs[idx])
        var_outputs = np2torch(outputs[idx])
        return var_inputs, var_outputs

    start = time.time()

    loss_all = 0
    loss_val = 1.
    loss_best = 1.
    stops = 0

    opt_enc = optim.SGD(encoder.parameters(), lr=lr)
    opt_dec = optim.SGD(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for i in range(1, niter + 1):
        opt_enc.zero_grad()
        opt_dec.zero_grad()

        var_inp, var_targ = get_batch(inputs_train, outputs_train, bsz)

        outs, attns = seq2seq_attn(var_inp, var_targ, encoder, decoder, True)
        loss = criterion(outs, var_targ)
        loss.backward()
        opt_enc.step()
        opt_dec.step()

        loss_all += loss.data[0] / args.future

        if i % iprint == 0:
            loss_avg = loss_all / iprint
            loss_all = 0
            loss_val = seq2seq_attn(inputs_valid, outputs_valid, encoder, decoder)
            print('iter: %d time: %s lr: %.5f train: %.4f loss: %.4f' % (
                i, Utils.timeSince(start, i / niter), lr, loss_avg, loss_val))
            # lr decay
            if loss_avg > loss_best:
                stops += 1
                if stops > 3:
                    if lr <= lr_min:
                        return
                    else:
                        lr /= lr_decay
                        stops = 0
            else:
                loss_best = loss_avg


trainIters(encoder, decoder,
           args.bsz, args.niter, args.iprint,
           args.lr, args.lr_min, args.lr_decay)

torch.save((encoder, decoder), args.savepath)
