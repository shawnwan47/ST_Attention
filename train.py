import time

import torch
import torch.nn as nn
import torch.optim as optim

from Utils import load_flow, split_dataset, timeSince, get_batch
from Models import EncoderRNN, AttnDecoderRNN
from Trainer import train_seq2seq_attn
from Constants import USE_CUDA


# CUDA
torch.manual_seed(47)
if torch.cuda.is_available() and USE_CUDA:
    torch.cuda.manual_seed(47)

# data
features, labels, days, times, flow_mean, flow_std = load_flow()
features_train, features_valid, features_test = split_dataset(features)
labels_train, labels_valid, labels_test = split_dataset(labels)


# model
ndim = features_train.shape[-1]
nhid = 256
encoder = EncoderRNN(ndim, nhid)
decoder = AttnDecoderRNN(ndim, nhid)

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# optimization
bsz = 10
niter = 10000
print_every = 100
lr = 0.01


# training
def trainIters(enc, dec, bsz=bsz, niter=niter, print_every=print_every, lr=lr):
    start = time.time()

    print_loss_all = 0  # Reset every print_every

    opt_enc = optim.Adam(enc.parameters(), lr=lr)
    opt_dec = optim.Adam(dec.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for iteration in range(1, niter + 1):
        print(iteration)
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

    torch.save(enc, 'enc.pk')
    torch.save(dec, 'dec.pk')


trainIters(encoder, decoder)
