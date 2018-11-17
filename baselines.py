import argparse
import pickle
import numpy as np

from lib import config
# from lib import Loss
from lib import utils
from lib import IO
from lib.Loader import get_loader


print('waht')
# args = argparse.ArgumentParser()
# config.add_data(args)
# print(args)
# args = args.parse_args()
# print(args)
# config.update_data(args)

# DATA
df = get_loader('BJ_highway').load_ts('15min')
data_train, data_valid, data_test, mean, std = IO.prepare_dataset(
    df, 60, 60
)
rescaler = utils.Rescaler(mean, std)

# MODEL
# loss = Loss.Loss(metrics=args.metrics, horizons=args.horizons)
