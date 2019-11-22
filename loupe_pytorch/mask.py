import torch
import torch.nn as nn

from . import layers

class Mask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode, is_epi, eps=0.01):
        super(Mask, self).__init__()
        
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.device = device
        self.sparsity = sparsity
        self.straight_through_mode = straight_through_mode
        self.sigmoid = nn.Sigmoid()
        
        # MaskNet
        if is_epi:
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0]))         
        else:
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
        self.pmask.requires_grad = True
        self.pmask.data.uniform_(eps, 1-eps)
        self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
        self.pmask.data = self.pmask.data.to(self.device)
        
    def squash_mask(self, mask):
        if len(mask.shape) == 1: # EPI mask
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, self.image_dims[1])
        return self.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty_like(self.pmask).uniform_(0, 1).to(self.device)
        if len(random_uniform.shape) == 1: # EPI mask
            random_uniform = random_uniform.unsqueeze(-1)
            random_uniform = random_uniform.expand(-1, self.image_dims[1])
        return self.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self, epoch=0, tot_epochs=0):
        # Apply probabilistic mask
        probmask = self.squash_mask(self.pmask)
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
        
class CondMask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode, eps=0.01):
        super(CondMask, self).__init__()
        
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.device = device
        self.sparsity = sparsity
        self.straight_through_mode = straight_through_mode
        self.sigmoid = nn.Sigmoid()
        
        # MaskNet outputs a vector of probabilities corresponding to image height
        self.fc1 = nn.Linear(1, self.image_dims[0])
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(self.image_dims[0], self.image_dims[0])

    def squash_mask(self, mask):
        # Takes in probability vector and outputs 2d probability mask  
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, self.image_dims[1])
        return self.sigmoid(self.pmask_slope*mask)
    
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
        random_uniform = torch.empty(mask.shape[0], self.image_dims[0]).uniform_(0, 1).to(self.device)
        random_uniform = random_uniform.unsqueeze(-1)
        random_uniform = random_uniform.expand(-1, -1, self.image_dims[1])
        return self.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self, condition, get_prob_mask=False, epoch=0, tot_epochs=0):
        fc_out = self.relu(self.fc1(condition))
        fc_out = self.fc_final(fc_out)

        # probmask is of shape (B, img_height)
        # Apply probabilistic mask
        probmask = self.squash_mask(fc_out)
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