import argparse

from lib import config
from lib import Loader




args = argparse.ArgumentParser()
config.add_data(args)
args = args.parse_args()
config.update_data(args)

assert args.dataset in ['BJ_highway', 'BJ_metro']

def load_od():
    if args.dataset == 'BJ_highway': loader = Loader.BJLoader('highway')
    else: loader = Loader.BJLoader('metro')
    od = loader.load_ts_od()
    hour = od.index.get_level_values(0)
    return od[(hour >= args.start) & (hour < args.end)]


def period_to_condition(idx, args.period, args.freq, args.start, args.end):
    if args.period == 'continous':
        ret = idx.week == idx.week[len(idx) // 2]
    if args.period == 'daily':
        for

def get_week(idx):
    return

def get_time(idx):
    return idx.time[idx.dayofyear == idx.dayofyear[0]]



def get_indexers(datetimeindex):




if __name__ == '__main__':
    od = load_od()
