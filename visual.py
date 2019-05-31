from config import Config
from lib import get_loader
from lib.plot import
import seaborn as sns


def load_ts(**kwargs):
    config = Config(**kwargs)
    config.set_dataset()
    ts = get_loader(config.dataset).load_ts()
    return ts


def plot_ts(**kwargs):
    # daily ts
    daily = ts[:config.num_times].sum(axis=1)
    plt.fiture()
    sns.ts_plot(daily)
    plt.savefig('daily.eps')
    print(daily.shape)
    weekly = ts[:config.num_times * 7].sum(axis=1)
    print(weekly.shape)
    sns.ts_plot(daily)


if __name__=='__main__':
    import fire
    fire.Fire()
