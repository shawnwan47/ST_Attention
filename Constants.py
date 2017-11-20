import sys

DATA_PATH = './data/'
FIG_PATH = './fig/'

DAYS = 22
WEEKDAY = 5  # (day - WEEKDAY) % 7
DAYS_TRAIN = 14
DAYS_TEST = 4
EPS = sys.float_info.epsilon

USE_CUDA = False
