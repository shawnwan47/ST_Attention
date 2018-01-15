# data as highway or metro
DATA_TYPE = 'metro'
DATA_FOLDER = DATA_TYPE + '/'
DATA_PATH = './data/' + DATA_FOLDER
FIG_PATH = './fig/' + DATA_FOLDER
MODEL_PATH = './model/' + DATA_FOLDER

if DATA_TYPE == 'highway':
    DIM = 284
    DAYS = 184
    DAYS_TRAIN = 120
    DAYS_TEST = 30
    WEEKDAY = 5  # (day - WEEKDAY) % 7
else:
    DIM = 538
    DAYS = 22
    DAYS_TRAIN = 14
    DAYS_TEST = 4
    WEEKDAY = 0

MAX_SEQ_LEN = 96 * 8
EPS = 1e-8
