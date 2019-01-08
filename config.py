import yaml
from constants import MODEL_PATH


class Config:
    def __init__(self, **kwargs):
        self.init_config()
        for k, v in kwargs.items():
            setattr(self, k, v)
        default = yaml.load(open('default.yaml'))
        self.set_data(default['data'])
        self.set_model(default['model'])

    def set_config(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k) or not getattr(self, k):
                setattr(self, k, v)

    def init_config(self):
        # data
        self.dataset = 'LA'
        self.bday = False
        self.start = 0
        self.end = 24
        # model
        self.path = MODEL_PATH
        self.model = 'STTransformer'
        self.dropout = 0.2

        self.batch_size = None
        self.paradigm = None
        self.hidden_size = None
        self.num_layers = None
        self.encoder_layers = None
        self.decoder_layers = None
        self.heads = None
        self.mask = None
        # cuda
        self.cuda = False
        self.gpuid = 3
        self.seed = 47
        # optim
        self.criterion = 'SmoothL1Loss'
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.epoches = 20

    def set_data(self, default):
        self.set_config(default['dataset'][self.dataset])
        self.set_config(default['task'][self.task])
        self.num_times = (self.end - self.start) * 60 // self.freq
        self.freq = str(self.freq) + 'min'

    def set_model(self, default):
        self.set_config(default['model'][self.model])
        self.set_config(default['paradigm'][self.paradigm])
        # model name
        self.path += self.dataset + '/'
        self.path += 'BDay' if self.bday else ''
