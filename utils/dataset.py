import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from utils.util import mu_law_encode, mu_law_decode 

class CustomerDataset(Dataset):
    def __init__(self,
                 path,
                 upsample_factor=120,
                 local_condition=True,
                 global_condition=False):

        self.path = path
        self.metadata = self.get_metadata(path)

        self.upsample_factor = upsample_factor

        self.local_condition = local_condition
        self.global_condition = global_condition

    def __getitem__(self, index):

        sample = np.load(os.path.join(self.path, 'audio', self.metadata[index]))
        condition = np.load(os.path.join(self.path, 'mel', self.metadata[index]))

        length = min([len(sample), len(condition) * self.upsample_factor])

        sample = sample[: length]
        condition = condition[: length // self.upsample_factor , :]
        
        sample = sample.reshape(-1, 1)

        if self.local_condition:
            return sample, condition
        else:
            return sample

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self, path):
        with open(os.path.join(path, 'names.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        return metadata

class CustomerCollate(object):

    def __init__(self,
                 upsample_factor=120,
                 condition_window=200,
                 local_condition=True,
                 global_condition=False):

        self.upsample_factor = upsample_factor
        self.condition_window = condition_window
        self.sample_window = condition_window * upsample_factor
        self.local_condition = local_condition
        self.global_condition = global_condition

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _collate_fn(self, batch):
        
        sample_batch = []
        condition_batch = []
        for (i, x) in enumerate(batch):
            if len(x[1]) < self.condition_window:
                sample = np.pad(x[0], [[0, self.sample_window - len(x[0])], [0, 0]], 'constant')
                condition = np.pad(x[1], [[0, self.condition_window - len(x[1])], [0, 0]], 'edge')
            else:
                lc_index = np.random.randint(0, len(x[1]) - self.condition_window)
                sample = x[0][lc_index * self.upsample_factor :
                    (lc_index + self.condition_window) * self.upsample_factor]
                condition = x[1][lc_index : (lc_index + self.condition_window)]
            sample_batch.append(sample)
            condition_batch.append(condition)
            
        sample_batch = np.stack(sample_batch)
        condition_batch = np.stack(condition_batch)
        sample_batch = mu_law_encode(sample_batch)
        sample_batch = mu_law_decode(sample_batch)
 
        samples = torch.FloatTensor(sample_batch).transpose(1, 2)
        conditions = torch.FloatTensor(condition_batch).transpose(1, 2)
 
        return samples, conditions
