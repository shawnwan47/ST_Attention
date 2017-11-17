from imp import reload
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
import models


parser = argparse.ArgumentParser(description='Traffic Flow Prediction Models')

parser.add_argument('--freq', type=int, default=15,
                    help='frequency of traffic flow')
parser.add_argument('--nprev', type=int, default=4,
                    help='number of previous flow')

parser.add_argument('--nemb_days', type=int, default=16,
                    help='size of day embeddings')
parser.add_argument('--nemb_time', type=int, default=16,
                    help='size of time embeddings')

parser.add_argument('--nhid_nn', type=int, default=256,
                    help='number of hidden units per layer')

parser.add_argument('--rnn', type=str, default='GRU',
                    help='RNN: (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid_rnn', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')


parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--lr_min', type=float, default=0.00001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay for optimizers')

parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--interval', type=int, default=1, metavar='N',
                    help='report interval')

parser.add_argument('--seed', type=int, default=47, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')

args = parser.parse_args()

print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have CUDA, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# DATA
###############################################################################
reload(utils)
# images
images, labels, mean, std = utils.load_flow_images(args.freq, args.nprev)
days_img, time_img = utils.load_daystime(args.freq, args.nprev)

# seqs
seqs, vals, mean, std = utils.load_flow_seqs(args.freq)
days_seq, time_seq = utils.load_daystime(args.freq)

if args.cuda:
    mean, std = mean.cuda(), std.cuda()


###############################################################################
# MODEL
###############################################################################
reload(models)
ndim = images[0].size(-1)
nin = ndim * args.nprev
nhid_nn = args.nhid_nn
nhid_rnn = args.nhid_rnn
nlayers = args.nlayers
embed_days = (7, args.nemb_days)
embed_time = (time_seq[0].max() + 1, args.nemb_time)

linear_regression = models.LR(nin, ndim)
neural_network = models.NN(nin, ndim, nhid_nn)
cnn = models.CNN(ndim, args.nprev, (3, 5), (1, 2))

embedding_nn = models.EmbeddingNN(nin, ndim, embed_days, embed_time, nhid_nn)
embedding_cnn = models.EmbeddingCNN(
    ndim, args.nprev, (3, 5), (1, 2), embed_days, embed_time)

rnn_tanh = models.RNN('RNN_TANH', ndim, nhid_rnn, nlayers)
gru = models.RNN('GRU', ndim, nhid_rnn, nlayers)
lstm = models.RNN('LSTM', ndim, nhid_rnn, nlayers)

embedding_rnn_tanh = models.EmbeddingRNN(
    'RNN_TANH', ndim, nhid_rnn, nlayers, embed_days, embed_time)
embedding_gru = models.EmbeddingRNN(
    'GRU', ndim, nhid_rnn, nlayers, embed_days, embed_time)
embedding_lstm = models.EmbeddingRNN(
    'LSTM', ndim, nhid_rnn, nlayers, embed_days, embed_time)

cnn_lstm = models.CRNN('LSTM', ndim, nhid_rnn, nlayers)

if args.cuda:
    linear_regression.cuda()
    neural_network.cuda()
    cnn.cuda()

    embedding_nn.cuda()
    embedding_cnn.cuda()

    rnn_tanh.cuda()
    gru.cuda()
    lstm.cuda()

    embedding_rnn_tanh.cuda()
    embedding_gru.cuda()
    embedding_lstm.cuda()


###############################################################################
# TRAIN
###############################################################################
mean = Variable(mean)
std = Variable(std)


def normalize(data):
    ret = data.add(-mean.expand_as(data)).div(std.expand_as(data))
    ret[ret != ret] = 0
    return ret


def denormalize(data):
    return data.mul(std.expand_as(data)).add(mean.expand_as(data))


def WAPE(x, y):
    return (x - y).abs().sum() / y.sum()


criterion = nn.L1Loss()
criterion = WAPE
epochs = args.epochs
interval = args.interval


def training(model, data_type, savepath, embedding=False):
    savepath += '_' + str(args.freq) + '.pt'
    if data_type == 'images':
        data_train, data_valid, data_test = images
        target_train, target_valid, target_test = labels
        days_train, days_valid, days_test = days_img
        time_train, time_valid, time_test = time_img
    else:
        data_train, data_valid, data_test = seqs
        target_train, target_valid, target_test = vals
        days_train, days_valid, days_test = days_seq
        time_train, time_valid, time_test = time_seq

    def predict(data, target, days, time):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
            days = days.cuda()
            time = time.cuda()
        if embedding:
            outputs = model(data, days, time)
        else:
            outputs = model(data)
        if data_type == 'images':
            outputs += data[:, 0, -1]
        else:
            outputs += data
            outputs = denormalize(outputs)
            target = denormalize(target)
        return criterion(outputs, target)

    print(89 * '=')
    print(savepath.upper())
    lr = args.lr
    if data_type == 'images':
        bsz = args.batch_size
        batches = data_train.size(0) // bsz
    else:
        bsz = 1
        batches = data_train.size(0)
    loss_best = None
    for epoch in range(epochs):
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.weight_decay)
        # shuffle data for each epoch
        loss_train = 0
        rand_idx = torch.randperm(data_train.size(0))
        for batch in range(batches):
            data = data_train[rand_idx][batch * bsz:(batch + 1) * bsz]
            target = target_train[rand_idx][batch * bsz:(batch + 1) * bsz]
            days = days_train[rand_idx][batch * bsz:(batch + 1) * bsz]
            time = time_train[rand_idx][batch * bsz:(batch + 1) * bsz]
            data = Variable(data)
            target = Variable(target)
            days = Variable(days)
            time = Variable(time)

            optimizer.zero_grad()
            loss = predict(data, target, days, time)
            loss.backward()
            if data_type == 'seqs':
                nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            loss_train += loss.data
        # turn on eval
        model.eval()

        data = Variable(data_valid, volatile=True)
        target = Variable(target_valid, volatile=True)
        days = Variable(days_valid, volatile=True)
        time = Variable(time_valid, volatile=True)
        loss_valid = predict(data, target, days, time)

        if (epoch + 1) % interval == 0:
            data = Variable(data_test, volatile=True)
            target = Variable(target_test, volatile=True)
            days = Variable(days_test, volatile=True)
            time = Variable(time_test, volatile=True)
            loss_test = predict(data, target, days, time)
            print("%d: lr %.4f, train %.4f, valid %.4f, test %.4f" % (
                epoch + 1,
                lr,
                loss_train[0] / batches,
                loss_valid.data[0],
                loss_test.data[0]))

        if not loss_best or loss_valid.data[0] < loss_best:
            loss_best = loss_valid.data[0]
            with open(savepath, 'wb') as f:
                torch.save(model, f)
        elif lr > args.lr_min:
            lr /= 2
        else:
            break


training(linear_regression, 'images', 'linear_regression')

training(neural_network, 'images', 'neural_network')
training(cnn, 'images', 'cnn')

training(embedding_nn, 'images', 'embedding_nn', embedding=True)
training(embedding_cnn, 'images', 'embedding_cnn', embedding=True)

training(rnn_tanh, 'seqs', 'rnn_tanh')
training(gru, 'seqs', 'gru')
training(lstm, 'seqs', 'lstm')

training(embedding_rnn_tanh, 'seqs', 'embedding_rnn_tanh', True)
training(embedding_gru, 'seqs', 'embedding_gru', True)
training(embedding_lstm, 'seqs', 'embedding_lstm', True)


###############################################################################
# TEST
###############################################################################
