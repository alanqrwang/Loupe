import torch
import torch.nn as nn
import numpy as np

from . import mask, straight_through_sample, utils, unet
from modl import unet_with_dc
import cascadenet_pytorch

import matplotlib.pyplot as plt

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

class Loupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, recon_type, is_epi, straight_through_mode='relax', K=5):
        super(Loupe, self).__init__()
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.device = device
        self.recon_type = recon_type

        # Mask
        self.mask = mask.Mask(image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode, is_epi)

        # Unet
        if recon_type == 'unet':
            self.recon = unet.Unet(image_dims)

        elif recon_type == 'cascade':
            linear = False
            self.recon = cascadenet_pytorch.model_pytorch.CascadeNet(linear, n_ch_out=image_dims[-1], nc=K) 

        elif recon_type == 'modl':
            # Modl from Jinwei
            self.modl = unet_with_dc.Unet_with_DC(input_channels=image_dims[-1],
                                          output_channels=image_dims[-1],
                                          num_filters=[2**i for i in range(5, 10)],
                                          lambda_dll2=0.001,
                                          K=K)

    def forward(self, inp, sample_mask=None):
        original_shape = inp.shape
        if inp.shape[-1] == 1:# 'data must have complex dimension'
            inp = torch.cat((inp, torch.zeros_like(inp)), dim=3)
        # input -> kspace via FFT
        ksp = utils.fft(inp)

        # build probability mask
        if sample_mask is not None:
            mask = sample_mask
        else:
            mask = self.mask()

        # Under-sample and back to image space via IFFT
        undersampled = undersample(ksp, mask)
        zf = utils.ifft(undersampled)

        zf = zf.permute(0, 3, 1, 2) # Reshape for convolution

        if self.recon_type == 'unet':
            unet_out = self.recon(zf)
            # Residual layer
            # Take magnitude if original input had one channel
            if original_shape[-1] == 1:
                zf = zf.norm(dim=1, keepdim=True)
            x = zf + unet_out

        elif self.recon_type == 'cascade': 
            mask_expand = mask.expand_as(zf)
            undersampled = undersampled.permute(0, 3, 1, 2)
            # print('mask', mask_expand.shape)
            # print('x', x.shape)
            # print('undersampled', undersampled.shape)
            x = self.recon(zf, undersampled, mask_expand)

        else:
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, self.image_dims[-1])
            mask = mask.expand(len(x), -1, -1, -1)
            mask = mask[:, None, ...]
            print('MASK', mask.shape)

            x = self.modl(zf, torch.ones_like(mask), mask)

        x = x.permute(0, 2, 3, 1) # Reshape for convolution

        
        return x

class CondLoupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, recon_type, straight_through_mode='relax'):
        super(CondLoupe, self).__init__()
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.device = device
        self.recon_type = recon_type

        # Mask
        self.mask = mask.CondMask(image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode)

        assert recon_type in ['unet', 'modl']
        # Unet
        self.unet = unet.Unet(image_dims)

        # Modl from Jinwei
        self.modl = unet_with_dc.Unet_with_DC(input_channels=image_dims[-1],
                                      output_channels=image_dims[-1],
                                      num_filters=[2**i for i in range(5, 10)],
                                      lambda_dll2=0.001,
                                      K=1)

    def forward(self, x, condition):
        if self.recon_type == 'unet':
            # hard-coded UNet
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

            x = x.permute(0, 3, 1, 2) # Reshape for convolution
            x = self.unet(x)
            unet_tensor = x.permute(0, 2, 3, 1) # Reshape for convolution

            # final output from model
            out = unet_tensor + abs_tensor
            return out

        else:
            # build probability mask
            mask = self.mask(condition)

            x = self.modl(x, torch.ones_like(mask), mask)

            return x



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

