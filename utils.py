# define dataset class to feed the model
import numpy as np 
import os
import sys
import time
import pandas as pd
import pickle
from ray.tune.registry import register_trainable

class ProgressBar():
    def __init__(self,worksum,info="",auto_display=True):
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display
    def startjob(self):
        self.begin_time = time.time()
    def complete(self,num):
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display == True:
            self.display_progress_bar()
    def display_progress_bar(self):
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
        str_log = ("%s %.2f %% %s %s/%s \t used:%ds eta:%d s" % (self.info,percent,strprogress,self.finishsum,self.worksum,self.gaptime,eta_time))
        sys.stdout.write('\r' + str_log)

def ma_sample(spaces):
    retval = {}
    for k,v in spaces.items():
        retval[k] = v.sample()
    return retval

def get_winrate_and_weight(logdir, league):
    """Restore checkpoint order and rewards from the headerless league table."""
    import ray
    from pathlib import Path
    table = pd.read_csv(Path(logdir) / 'winrates.csv', header=None)
    rewards = dict(zip(table.iloc[0, 1:], table.iloc[1, 1:].astype(float)))
    paths = sorted((Path(logdir) / 'weights').glob('c_*.pkl'),
                   key=lambda path: int(path.stem.split('_')[-1]))
    if not paths:
        raise ValueError('No league checkpoints found')
    winrates = []
    for path in paths:
        if path.stem not in rewards:
            raise ValueError('Missing league statistics for ' + path.stem)
        with path.open('rb') as stream:
            weight = pickle.load(stream)
        ray.get(league.add_weight.remote(weight))
        winrates.append(rewards[path.stem])
    ray.get(league.set_winrates.remote(winrates))
    # The learner can be newer than the last historical opponent snapshot.
    final_path = Path(logdir) / 'output_weight.pkl'
    if final_path.exists():
        with final_path.open('rb') as stream:
            return pickle.load(stream)
    return weight


def save_learner_weights(output_dir, weight):
    """Replace the latest learner snapshot only after its complete write."""
    from pathlib import Path
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'output_weight.pkl.tmp'
    with temporary.open('wb') as stream:
        pickle.dump(weight, stream)
    temporary.replace(directory / 'output_weight.pkl')
