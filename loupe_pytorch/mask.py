import torch
import torch.nn as nn

class Mask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, eps=0.01, slope=10):
        super(Mask, self).__init__()
        self.image_dims = image_dims
        self.sparsity = sparsity
        self.device = device
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope

        self.weight = nn.Parameter(torch.FloatTensor(image_dims[0], image_dims[1], 2)) # Mask is same dimension as image plus complex domain
        self.weight.requires_grad = True
        self.weight.data.uniform_(eps, 1-eps)
        self.weight.data = -torch.log(1. / self.weight.data - 1.) / self.pmask_slope
        self.weight.data = self.weight.data.to(self.device)

    def sparsify(self, pmask):
        xbar = pmask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        sparse_pmask = le * pmask * r + (1-le) * (1 - (1 - pmask) * beta) 
        return sparse_pmask

    def threshold(self, sparse_pmask):
        random_uniform = torch.empty(*self.image_dims, 2).uniform_(0, 1).to(self.device)
        return torch.sigmoid(self.sample_slope*(sparse_pmask - random_uniform))

    def undersample(self, x, prob_mask):
        mask = prob_mask.expand(x.shape[0], -1, -1, -1)
        x[:,:,:,0] = torch.mul(x[:,:,:,0], mask[:,:,:,0])
        x[:,:,:,1] = torch.mul(x[:,:,:,1], mask[:,:,:,0])
        return x

    def forward(self, x):
        pmask = torch.sigmoid(self.pmask_slope*self.weight.data)
        sparse_pmask = self.sparsify(pmask)
        threshold_mask = self.threshold(sparse_pmask)
        x = self.undersample(x, threshold_mask)
        return x