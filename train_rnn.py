import torch
from torch.autograd import Variable

from Config import Config
import Models
import Optim
import Loss
import Utils


config = Config('Seq2Seq')
config.add_train()
config.add_rnn()
config.add_attention()
config.add_optim()
args = config.parse_args()
if torch.cuda.is_available() and not args.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")
if args.gpuid:
    torch.cuda.set_device(args.gpuid[0])
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)

# DATA
(inputs_train, inputs_valid, _,
 targets_train, targets_valid, _,
 daytimes_train, daytimes_valid, _,
 flow_mean, flow_std) = Utils.load_data(args)
inputs_train = Variable(inputs_train)
targets_train = Variable(targets_train)
inputs_valid = Variable(inputs_valid, volatile=True)
targets_valid = Variable(targets_valid, volatile=True)
flow_mean = Variable(flow_mean, requires_grad=False)
flow_std = Variable(flow_std, requires_grad=False)


def denormalize(flow):
    return flow * flow_std + flow_mean


# MODEL
print("Model: %s" % (Utils.modelpath(args)))
model = Models.Seq2Seq(args)
model = model.cuda() if args.gpuid else model

# LOSS
criterion = getattr(Loss, args.loss)

# OPTIM
optimizer = Optim.Optim(args)
optimizer.set_parameters(model.parameters())

# training
for epoch in range(args.nepoch):
    # train for random day
    loss_train = []
    for d in torch.randperm(inputs_train.size(1)):
        # data definition
        src = inputs_train[:args.past, d].unsqueeze(1)
        tgt = targets_train[args.past:, d].unsqueeze(1)
        inputs = inputs_train[args.past:, d].unsqueeze(1)
        # model loss
        outputs = model(src, inputs, teach=True)
        outputs = outputs[0]
        outputs = denormalize(outputs)
        tgt = denormalize(tgt)
        loss = criterion(outputs, tgt)
        # optimization
        loss.backward()
        optimizer.step()
        # add loss
        loss_train.append(loss.data[0])
    loss_train = sum(loss_train) / len(loss_train)

    # valid for every time
    loss_valid = []
    for t in range(args.past, inputs_valid.size(0) - args.future):
        src = inputs_valid[:t]
        tgt = targets_valid[t:t + args.future]
        inputs = inputs_valid[t:t + args.future]
        outputs = model(src, inputs, teach=False)
        outputs = outputs[0]
        outputs = denormalize(outputs)
        tgt = denormalize(tgt)
        loss = criterion(outputs, tgt)
        loss_valid.append(loss.data[0])
    loss_valid = sum(loss_valid) / len(loss_valid)

    # update
    print('epoch: %d loss:%s train: %.4f valid: %.4f' % (
        epoch, args.loss, loss_train, loss_valid))

    optimizer.updateLearningRate(loss_valid)

torch.save(model.cpu(), Utils.modelpath(args))
