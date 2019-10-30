import torch
import torch.nn as nn

from loupe_pytorch import straight_through_sample

class Unet(nn.Module):
    def __init__(self, image_dims):
        super(Unet, self).__init__()
        
        # UNet
        self.dconv_down1 = self.double_conv(2, 64)
        self.dconv_down2 = self.double_conv(64, 128)
        self.dconv_down3 = self.double_conv(128, 256)
        self.dconv_down4 = self.double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(256 + 512, 256)
        self.dconv_up2 = self.double_conv(128 + 256, 128)
        self.dconv_up1 = self.double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, image_dims[-1], 1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        return x
    
class Mask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, eps=0.01):
        super(Mask, self).__init__()
        
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.device = device
        self.sparsity = sparsity
        
        # MaskNet
        # Mask is same dimension as image
        self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
        self.pmask.requires_grad = True
        self.pmask.data.uniform_(eps, 1-eps)
        self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
        self.pmask.data = self.pmask.data.to(self.device)
        
    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty(*mask.shape).uniform_(0, 1).to(self.device)
        return torch.sigmoid(self.sample_slope*(mask - random_uniform))
    
    def forward(self, mode, epoch=0, tot_epochs=0):
        # Apply probabilistic mask
        probmask = self.squash_mask(self.pmask)
        # Sparsify
        sparse_mask = self.sparsify(probmask)
        # Threshold
        assert mode in ['ste-identity', 'ste-sigmoid-fixed', 'ste-sigmoid-anneal', 'relax'], \
                       'mode should be ste-identity, ste-sigmoid-fixed, ste-sigmoid-anneal, relax'
        if mode == 'ste-identity':
            stidentity = straight_through_sample.STIdentity.apply
            mask = stidentity(sparse_mask)
        elif mode == 'ste-sigmoid':
            stsigmoid = straight_through_sample.STSigmoid.apply
            mask = stsigmoid(sparse_mask, epoch, tot_epochs)
        else:
            mask = self.threshold(sparse_mask)
        return mask
    
def complex_abs(x):
    # Tensor should be of shape (N, l, w, 2)
    x = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2).unsqueeze(3)
    return x

def undersample(x, mask):
    # Expects shape of mask to be (L, W)
    mask = mask.expand(x.shape[0], -1, -1)
    undersampled_x = torch.zeros_like(x)
    undersampled_x[:,:,:,0] = torch.mul(x[:,:,:,0], mask[:,:,:])
    undersampled_x[:,:,:,1] = torch.mul(x[:,:,:,1], mask[:,:,:])
    return undersampled_x

class Loupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device):
        super(Loupe, self).__init__()
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        self.device = device

        # Mask
        self.mask = Mask(image_dims, pmask_slope, sample_slope, sparsity, device)

        # Unet
        self.unet = Unet(image_dims)

    def forward(self, x, mode='relax'):
        # if necessary, concatenate with zeros for FFT
        if x.shape[-1] == 1:
            x = torch.cat((x, torch.zeros_like(x)), dim=3)

        # input -> kspace via FFT
        x = torch.fft(x, signal_ndim=2)

        # build probability mask
        mask = self.mask(mode)
        
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