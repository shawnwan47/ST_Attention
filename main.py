import numpy as np

import torch
import torch.optim as optim
import fire

from config import Args
from lib import utils
from lib import pt_utils
from lib import Loss
from lib import Trainer

from models import builder

args = Args()
print(args)

class Main:
    def __init__(self, *args):
        self.config = Args(*args)
        # CUDA
        self.config.cuda &= torch.cuda.is_available()
        if self.config.cuda:
            torch.cuda.set_device(self.config.gpuid)
            print(f'Using GPU: {self.config.gpuid}')
        else:
            print('Using CPU')
        if self.config.seed is not None:
            if self.config.cuda:
                torch.cuda.manual_seed(self.config.seed)
            else:
                torch.manual_seed(self.config.seed)

        self.loader_train, self.loader_valid, self.loader_test, mean, std = pt_utils.load_loaders(self.config)

        self.model = builder.build(args, mean, std)
        print(f'{args.path} parameters: {pt_utils.count_parameters(model)}')
        if args.cuda:
            self.model.cuda()
        self.criterion = getattr(torch.nn, args.criterion)()

    @torch.no_grad()
    def test():
        model.load_state_dict(torch.load(args.path))
        self.model.eval()

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
    fire.Fire()
