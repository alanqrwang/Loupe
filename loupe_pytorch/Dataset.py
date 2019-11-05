import torch
from torch.utils import data

class CondDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels, conditions):
        self.labels = labels
        self.data = data
        self.conditions = conditions

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        c = self.conditions[index]

        return X, y, c
