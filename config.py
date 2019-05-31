import yaml
from constants import CONFIG, DATASETS
from constants import DATASET_PATH, MODEL_PATH, RESULT_PATH, FIG_PATH
from pathlib import Path


class Config:
    def __init__(self, **kwargs):
        self.init_config()
        self.default = yaml.load(open(CONFIG))
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_config(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k) or not getattr(self, k):
                setattr(self, k, v)

    def init_config(self):
        # data
        self.dataset = None
        self.bday = False
        self.start = None
        self.end = None
        self.freq = None
        self.horizon = None
        self.horizons = None
        # model
        self.paradigm = None
        self.model = None
        self.mask = False
        # cuda
        self.cuda = False
        self.gpuid = 0
        self.seed = 47
        # optim
        self.criterion = 'L1Loss'
        self.dropout = 0.1
        self.weight_decay = 1e-5
        self.batch_size = None
        self.epoches = None
        self.patience = None
        # path
        self.path_dataset = Path(DATASET_PATH)
        self.path_model = Path(MODEL_PATH)
        self.path_result = Path(RESULT_PATH)
        self.path_fig = Path(FIG_PATH)

    def set_dataset(self):
        assert self.dataset in DATASETS
        default = self.default['data']
        self.path_dataset /= self.dataset
        self.path_model /= self.dataset
        self.path_result /= self.dataset
        self.path_fig /= self.dataset
        self.path_dataset.mkdir(parents=True, exist_ok=True)
        self.path_model.mkdir(parents=True, exist_ok=True)
        self.path_result.mkdir(parents=True, exist_ok=True)
        self.path_fig.mkdir(parents=True, exist_ok=True)
        self.set_config(default['dataset'][self.dataset])
        self.set_config(default['task'][self.task])
        self.num_times = (self.end - self.start) * 60 // self.freq
        self.freq = str(self.freq) + 'min'

    def set_model(self):
        assert self.model is not None
        default = self.default['model']
        self.set_config(default['model'][self.model])
        self.set_config(default['paradigm'][self.paradigm])
        # model name
        model_name = self.model
        for key in default['model'][self.model]:
            model_name += key + str(getattr(self, key))
        self.path_dataset /= model_name
        self.path_model /= model_name
        self.path_result /= model_name
        self.path_fig /= model_name
