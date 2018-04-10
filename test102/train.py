import utils
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CornellDataset(Dataset):
    '''What a dataset'''

    def __init__(self, datapoints :list):
        '''datapoints : `utils.prepareDataPoints`
        '''
        self.datapoints = datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx].X, self.datapoints[idx].Y