import torch
import torch.nn as nn

from . import unet, mask, straight_through_sample
    
def complex_abs(x):
    # Tensor should be of shape (N, l, w, 2)
    x = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2).unsqueeze(3)
    return x

def undersample(x, mask):
    if len(mask.shape) == 2: # This will be the case for normal Loupe
        mask = mask.expand(x.shape[0], -1, -1)
    undersampled_x = torch.zeros_like(x)
    undersampled_x[:,:,:,0] = torch.mul(x[:,:,:,0], mask)
    undersampled_x[:,:,:,1] = torch.mul(x[:,:,:,1], mask)
    return undersampled_x

class CondLoupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode='relax'):
        super(CondLoupe, self).__init__()
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.device = device

        # Mask
        self.mask = mask.CondMask(image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode)

        # Unet
        self.unet = unet.Unet(image_dims)

    def forward(self, x, condition):
        # if necessary, concatenate with zeros for FFT
        if x.shape[-1] == 1:
            x = torch.cat((x, torch.zeros_like(x)), dim=3)

        # input -> kspace via FFT
        x = torch.fft(x, signal_ndim=2)

        # build probability mask
        mask = self.mask(condition)
        
        # Under-sample and back to image space via IFFT
        x = undersample(x, mask)
        x = torch.ifft(x, signal_ndim=2).to(self.device)

        # Complex absolute layer
        abs_tensor = complex_abs(x)

        # hard-coded UNet
        x = x.view(-1, 2, self.image_dims[0], self.image_dims[1]) # Reshape for convolution
        x = self.unet(x)
        unet_tensor = x.view(-1, self.image_dims[0], self.image_dims[1], 2) # Reshape for convolution

        # final output from model
        out = unet_tensor + abs_tensor

        return out

class UnetLoupe(nn.Module):
    def __init__(self, image_dims, sample_slope, device, sample_mask):
        super(UnetLoupe, self).__init__()
        self.image_dims = image_dims
        self.sample_slope = sample_slope
        self.device = device

        # Fixed Mask
        self.sample_mask = sample_mask

        # Unet
        self.unet = Unet(image_dims)

            
    def forward(self, x, mode='relax'):
        # if necessary, concatenate with zeros for FFT
        if x.shape[-1] == 1:
            x = torch.cat((x, torch.zeros_like(x)), dim=3)

        # input -> kspace via FFT
        x = torch.fft(x, signal_ndim=2)

        # Under-sample and back to image space via IFFT
        x = undersample(x, self.sample_mask)
        x = torch.ifft(x, signal_ndim=2).to(self.device)

        # Complex absolute layer
        abs_tensor = complex_abs(x)

        # hard-coded UNet
        x = x.view(-1, 2, self.image_dims[0], self.image_dims[1]) # Reshape for convolution
        x = self.unet(x)
        unet_tensor = x.view(-1, self.image_dims[0], self.image_dims[1], 2) # Reshape for convolution

        # final output from model
        out = unet_tensor + abs_tensor

        return out
