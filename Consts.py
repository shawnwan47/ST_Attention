# data as highway or metro
DATA_TYPE = 'highway'
DATA_FOLDER = DATA_TYPE + '/'
DATA_PATH = './data/' + DATA_FOLDER
FIG_PATH = './fig/' + DATA_FOLDER
MODEL_PATH = './model/' + DATA_FOLDER

if DATA_TYPE == 'highway':
    DIM = 284
    DAYS = 184
    DAYS_TRAIN = 120
    DAYS_TEST = 30
else:
    DIM = 538
    DAYS = 22
    DAYS_TRAIN = 14
    DAYS_TEST = 4

MAX_SEQ_LEN = 96 * 8
EPS = 1e-8
