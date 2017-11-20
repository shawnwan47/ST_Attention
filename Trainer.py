from __future__ import division
from __future__ import print_function

import time

import torch
import torch.optim as optim
from torch.autograd import Variable


def train(var_input, var_target, encoder, decoder, opt_enc, opt_dec, criterion):
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    opt_enc.zero_grad()
    opt_dec.zero_grad()

    len_input = var_input.size()[0]
    len_target = var_target.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(len_input):
        encoder_output, encoder_hidden = encoder(
            var_input[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(len_target):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, var_target[di])
            decoder_input = var_target[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(len_target):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, var_target[di])
            if ni == EOS_token:
                break

    loss.backward()

    opt_enc.step()
    opt_dec.step()

    return loss.data[0] / len_target


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, lr=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    opt_enc = optim.SGD(encoder.parameters(), lr=lr)
    opt_dec = optim.SGD(decoder.parameters(), lr=lr)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        var_input = training_pair[0]
        var_target = training_pair[1]

        loss = train(var_input, var_target, encoder,
                     decoder, opt_enc, opt_dec, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
