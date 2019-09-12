import torch
import torch.nn as nn
from mask import Mask
from utils import plot_figure

class Loupe(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, base_size=64, kernel_size=3):
        super(Loupe, self).__init__()
        self.image_dims = image_dims
        self.device = device
        self.sparsity = sparsity

        self.conv1a = nn.Conv2d(2, base_size, kernel_size, padding=1) # 
        self.batchNorm1a = nn.BatchNorm2d(base_size)
        self.conv1b = nn.Conv2d(base_size, base_size, kernel_size, padding=1)
        self.batchNorm1b = nn.BatchNorm2d(base_size)
        self.avgpool1 = nn.AvgPool2d(2) # 64 x 128 x 128

        self.conv2a = nn.Conv2d(base_size, base_size*2, kernel_size, padding=1)
        self.batchNorm2a = nn.BatchNorm2d(base_size*2)
        self.conv2b = nn.Conv2d(base_size*2, base_size*2, kernel_size, padding=1)
        self.batchNorm2b = nn.BatchNorm2d(base_size*2)
        self.avgpool2 = nn.AvgPool2d(2) # 128 x 64 x 64 

        self.conv3a = nn.Conv2d(base_size*2, base_size*4, kernel_size, padding=1)
        self.batchNorm3a = nn.BatchNorm2d(base_size*4)
        self.conv3b = nn.Conv2d(base_size*4, base_size*4, kernel_size, padding=1)
        self.batchNorm3b = nn.BatchNorm2d(base_size*4)
        self.avgpool3 = nn.AvgPool2d(2) # 256 x 32 x 32

        self.conv4a = nn.Conv2d(base_size*4, base_size*8, kernel_size, padding=1)
        self.batchNorm4a = nn.BatchNorm2d(base_size*8)
        self.conv4b = nn.Conv2d(base_size*8, base_size*8, kernel_size, padding=1)
        self.batchNorm4b = nn.BatchNorm2d(base_size*8)
        self.avgpool4 = nn.AvgPool2d(2) # 512 x 16 x 16

        self.conv5a = nn.Conv2d(base_size*8, base_size*16, kernel_size, padding=1)
        self.batchNorm5a = nn.BatchNorm2d(base_size*16)
        self.conv5b = nn.Conv2d(base_size*16, base_size*16, kernel_size, padding=1)
        self.batchNorm5b = nn.BatchNorm2d(base_size*16)
        self.upsample5 = nn.Upsample(scale_factor=2) # 1024 x 32 x 32

        self.conv6a = nn.Conv2d(base_size*16 + base_size*8, base_size*8, kernel_size, padding=1)
        self.batchNorm6a = nn.BatchNorm2d(base_size*8)
        self.conv6b = nn.Conv2d(base_size*8, base_size*8, kernel_size, padding=1)
        self.batchNorm6b = nn.BatchNorm2d(base_size*8)
        self.upsample6 = nn.Upsample(scale_factor=2)

        self.conv7a = nn.Conv2d(base_size*8 + base_size*4, base_size*4, kernel_size, padding=1)
        self.batchNorm7a = nn.BatchNorm2d(base_size*4)
        self.conv7b = nn.Conv2d(base_size*4, base_size*4, kernel_size, padding=1)
        self.batchNorm7b = nn.BatchNorm2d(base_size*4)
        self.upsample7 = nn.Upsample(scale_factor=2)

        self.conv8a = nn.Conv2d(base_size*4 + base_size*2, base_size*2, kernel_size, padding=1)
        self.batchNorm8a = nn.BatchNorm2d(base_size*2)
        self.conv8b = nn.Conv2d(base_size*2, base_size*2, kernel_size, padding=1)
        self.batchNorm8b = nn.BatchNorm2d(base_size*2)
        self.upsample8 = nn.Upsample(scale_factor=2)

        self.conv9a = nn.Conv2d(base_size*2 + base_size, base_size, kernel_size, padding=1)
        self.batchNorm9a = nn.BatchNorm2d(base_size)
        self.conv9b = nn.Conv2d(base_size, base_size, kernel_size, padding=1)
        self.batchNorm9b = nn.BatchNorm2d(base_size)
        self.convFinal = nn.Conv2d(base_size, 1, 1)

        self.leakyReLU = nn.LeakyReLU()

        self.mask = Mask(image_dims, pmask_slope, sample_slope, sparsity, self.device)

    def A(self, x):
        conv1 = self.conv1a(x)
        conv1 = self.leakyReLU(conv1)
        conv1 = self.batchNorm1a(conv1)
        conv1 = self.conv1b(conv1)
        conv1 = self.leakyReLU(conv1)
        conv1 = self.batchNorm1b(conv1)
        
        pool1 = self.avgpool1(conv1)
        
        conv2 = self.conv2a(pool1)
        conv2 = self.leakyReLU(conv2)
        conv2 = self.batchNorm2a(conv2)
        conv2 = self.conv2b(conv2)
        conv2 = self.leakyReLU(conv2)
        conv2 = self.batchNorm2b(conv2)
        
        pool2 = self.avgpool2(conv2)
        conv3 = self.conv3a(pool2)
        conv3 = self.leakyReLU(conv3)
        conv3 = self.batchNorm3a(conv3)
        conv3 = self.conv3b(conv3)
        conv3 = self.leakyReLU(conv3)
        conv3 = self.batchNorm3b(conv3)
        
        pool3 = self.avgpool3(conv3)
        
        conv4 = self.conv4a(pool3)
        conv4 = self.leakyReLU(conv4)
        conv4 = self.batchNorm4a(conv4)
        conv4 = self.conv4b(conv4)
        conv4 = self.leakyReLU(conv4)
        conv4 = self.batchNorm4b(conv4)
        
        pool4 = self.avgpool4(conv4) # 512 x 16 x 16

        conv5 = self.conv5a(pool4)
        conv5 = self.leakyReLU(conv5)
        conv5 = self.batchNorm5a(conv5)
        conv5 = self.conv5b(conv5)
        conv5 = self.leakyReLU(conv5)
        conv5 = self.batchNorm5b(conv5)

        sub1 = self.upsample5(conv5) # 1024 x 32 x 32
        concat1 = torch.cat([conv4,sub1], dim=1)
        
        conv6 = self.conv6a(concat1)
        conv6 = self.leakyReLU(conv6)
        conv6 = self.batchNorm6a(conv6)
        conv6 = self.conv6b(conv6)
        conv6 = self.leakyReLU(conv6)
        conv6 = self.batchNorm6b(conv6)

        sub2 = self.upsample6(conv6)
        concat2 = torch.cat([conv3,sub2], dim=1)
        
        conv7 = self.conv7a(concat2)
        conv7 = self.leakyReLU(conv7)
        conv7 = self.batchNorm7a(conv7)
        conv7 = self.conv7b(conv7)
        conv7 = self.leakyReLU(conv7)
        conv7 = self.batchNorm7b(conv7)

        sub3 = self.upsample7(conv7)
        concat3 = torch.cat([conv2,sub3], dim=1)
        
        conv8 = self.conv8a(concat3)
        conv8 = self.leakyReLU(conv8)
        conv8 = self.batchNorm8a(conv8)
        conv8 = self.conv8b(conv8)
        conv8 = self.leakyReLU(conv8)
        conv8 = self.batchNorm8b(conv8)

        sub4 = self.upsample8(conv8)
        concat4 = torch.cat([conv1,sub4], dim=1)
        
        conv9 = self.conv9a(concat4)
        conv9 = self.leakyReLU(conv9)
        conv9 = self.batchNorm9a(conv9)
        conv9 = self.conv9b(conv9)
        conv9 = self.leakyReLU(conv9)
        conv9 = self.batchNorm9b(conv9)
        conv9 = self.convFinal(conv9)
        
        return conv9 

    def forward(self, x, batch_idx):
        # FFT into k-space
        x = x.view(-1, *self.image_dims, 1)
        x = torch.cat((x, torch.zeros(x.shape).to(self.device)), dim=3)
        x = torch.fft(x, signal_ndim=2)

        # Apply probabilistic mask
        x, mask = self.mask(x)

        # iFFT into image space
        x = torch.ifft(x, signal_ndim=2)

        x = x.view(-1, 2, *self.image_dims) # Reshape for convolution
        x = self.A(x)
        x = x.view(-1, *self.image_dims, 1) # Reshape for convolution
        x = x + self.complex_abs(x)

        return x, mask

    def complex_abs(self, x):
        # Tensor should be of shape (N, l, w, 2)
        x = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2)
        x = x.unsqueeze(3)
        return x # Append a new axis for purposes of shape consistency