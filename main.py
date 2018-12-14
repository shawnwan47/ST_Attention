import numpy as np

import torch
import torch.optim as optim

from config import Args
from lib import utils
from lib import pt_utils
from lib import Loss
from lib import Trainer

from models import builder

args = Args()
print(args)

# CUDA
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpuid)
    torch.cuda.manual_seed(args.seed)
    print(f'Using GPU: {args.gpuid}')
else:
    torch.manual_seed(args.seed)
    print('Using CPU')

# DATA
loader_train, loader_valid, loader_test, mean, std = pt_utils.load_loaders(args)

# MODEL
model = builder.build_model(args, mean, std)
if args.test:
    model.load_state_dict(torch.load(args.path + '.pt'))
if args.cuda:
    model.cuda()
print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')

# criterion, loss
criterion = getattr(torch.nn, args.criterion)()
loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)

# optimizer, scheduler
optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

# TRAINER
trainer = Trainer.Trainer(
    model=model,
    framework=args.framework,
    rescaler=rescaler,
    criterion=criterion,
    loss=loss,
    optimizer=optimizer,
    epoches=args.epoches,
    iterations=args.iterations,
    cuda=args.cuda
)


if not args.test:
    trainer.run(loader_train, loader_valid)
    torch.save(trainer.model.state_dict(), args.path + '.pt')

error_test = trainer.run_epoch(loader_test)
print(f'{args.path}:\n{error_test}')


if __name__ == '__main__':
    import fire
    fire.Fire()
