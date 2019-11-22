import torch.nn as nn

class LocalConnected2d(nn.Module):
    def __init__(self, in_features):
        super(LocalConnected2d, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_features)) 
        self.biases = nn.Parameter(torch.FloatTensor(in_features))
        
    def forward(self, x):
        return x * self.weights + self.biases