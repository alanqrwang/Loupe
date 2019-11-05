import torch
import torch.nn as nn

class LocalConnected2d(nn.Module):
    def __init__(self, in_features):
        super(LocalConnected2d, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_features)) 
        self.biases = nn.Parameter(torch.FloatTensor(in_features))
        
    def forward(self, x):
        return x * self.weights + self.biases

class CondMask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode, eps=0.01):
        super(CondMask, self).__init__()
        
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.device = device
        self.sparsity = sparsity
        assert straight_through_mode in ['ste-identity', 'ste-sigmoid-fixed', 'ste-sigmoid-anneal', 'relax'], \
                       'mode should be ste-identity, ste-sigmoid-fixed, ste-sigmoid-anneal, relax'
        self.straight_through_mode = straight_through_mode
        
        # MaskNet
        self.fc1 = nn.Linear(1, self.image_dims[0]*self.image_dims[1])
        self.relu = nn.ReLU()
        self.local1 = LocalConnected2d(self.image_dims[0]*self.image_dims[1])

    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        mask_out = torch.zeros_like(mask)
        xbar = mask.mean(-1).mean(-1)
        r = self.sparsity / xbar
        r = r.view(-1, 1, 1)
        beta = (1-self.sparsity) / (1-xbar)
        beta = beta.view(-1, 1, 1)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty_like(mask).uniform_(0, 1).to(self.device)
        return torch.sigmoid(self.sample_slope*(mask - random_uniform))
    
    def forward(self, condition, epoch=0, tot_epochs=0):
        fc_out = self.relu(self.fc1(condition))
        probmask = self.local1(fc_out)

        probmask = probmask.view(len(probmask), self.image_dims[0], self.image_dims[1])
        # Apply probabilistic mask
        probmask = self.squash_mask(probmask)
        # Sparsify
        sparse_mask = self.sparsify(probmask)
        # Threshold
        if self.straight_through_mode == 'ste-identity':
            stidentity = straight_through_sample.STIdentity.apply
            mask = stidentity(sparse_mask)
        elif self.straight_through_mode == 'ste-sigmoid':
            stsigmoid = straight_through_sample.STSigmoid.apply
            mask = stsigmoid(sparse_mask, epoch, tot_epochs)
        else:
            mask = self.threshold(sparse_mask)
        return mask