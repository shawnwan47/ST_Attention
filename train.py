import time
import argparse

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim

from Utils import load_flow, split_dataset, timeSince, get_batch
from Models import EncoderRNN, AttnDecoderRNN
from Trainer import train_seq2seq_attn
from Constants import USE_CUDA
from Loss import WAPE, MAPE

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# data
parser.add_argument('--granularity', type=int, default=15)
parser.add_argument('--history', type=int, default=40)
parser.add_argument('--future', type=int, default=8)
# model
# train
parser.add_argument('--bsz', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
parser.add_argument('--nepoch', type=int, default=20)
parser.add_argument('--nepoch_decay', type=int, default=10)
parser.add_argument('--niter', type=int, default=10000)
parser.add_argument('--num_iters_decay', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=16)

opt = parser.parse_args()

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


# data
features, labels, days, times, flow_mean, flow_std = load_flow()
features_train, features_valid, features_test = split_dataset(features)
labels_train, labels_valid, labels_test = split_dataset(labels)


# model
ndim = features_train.shape[-1]
nhid = 1024
encoder = EncoderRNN(ndim, nhid)
decoder = AttnDecoderRNN(ndim, nhid)

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# optimization
bsz = 100
niter = 10000
print_every = 100
lr = 0.01


# training
def trainIters(enc, dec, bsz, niter, print_every, lr=lr):
    start = time.time()

    print_loss_all = 0  # Reset every print_every

    opt_enc = optim.SGD(enc.parameters(), lr=lr)
    opt_dec = optim.SGD(dec.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for iteration in range(1, niter + 1):
        var_inp, var_targ = get_batch(features_train, labels_train, bsz)

        loss = train_seq2seq_attn(var_inp, var_targ, enc, dec,
                                  opt_enc, opt_dec, criterion)
        print_loss_all += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_all / print_every
            print_loss_all = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / niter),
                                         iteration, iteration / niter * 100,
                                         print_loss_avg))



trainIters(encoder, decoder)

torch.save(enc, 'enc.pk')
torch.save(dec, 'dec.pk')