import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvBlock(nn.Module):
    """ A Conv-Relu-BatchNorm block """
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0):
        
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_channels))

        self.basic_conv = nn.Sequential(*layers)   

    def forward(self, x):
        return self.basic_conv(x)

class ConvTransposeBlock(nn.Module):
    """ A ConvTranspose-Relu-BatchNorm block """
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0):
        
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_channels))

        self.basic_conv = nn.Sequential(*layers)   

    def forward(self, x):
        return self.basic_conv(x)

class Generator(nn.Module):
    """ A fully Convolutional Neural Network to generate images from random noise z """
    def __init__(self, nc, nin_w, nin_h, nout_w, nout_h):
        super().__init__()

        channels = [32, 64, 128, 64, 32, 1]
        kernels = [7, 5, 2, 5, 3, 3]        
        layers = []

        ic = nc
        padding = 0
        stride = 1

        for idx in range(len(channels)):
            # add a conv, followed by a sigmoid
            if idx == len(channels) - 1:
                layers.append(nn.ConvTranspose2d(ic, channels[idx], kernels[idx]))
            else:
                layers.append(ConvTransposeBlock(ic, channels[idx], kernels[idx]))
                ic = channels[idx]
        
        self.generate = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.generate(x))

class Discriminator(nn.Module):
    """ A Fully Convolutional neural net classifier """
    def __init__(self, nc, nw, nh, nclasses):
        super().__init__()
        channels = [64, 128, 256, 512]
        kernels = [7, 3, 3, 3]
        layers = []

        ic = nc
        wsize = nw
        hsize = nh
        padding = 0
        stride = 1

        for out_channel, kernel_size in zip(channels, kernels):
            layers.append(BasicConvBlock(ic, out_channel,
                kernel_size))
            ic = out_channel
            wsize = (wsize - kernel_size + 2*padding)//stride + 1
            hsize = (hsize - kernel_size + 2*padding)//stride + 1
        
        self.conv_block = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(ic, nclasses, (hsize, wsize))


    def forward(self, x):
        final_conv = self.conv_block(x)
        class_scores = self.classifier(final_conv)
        class_scores = torch.reshape(class_scores, (-1, 1))

        return class_scores
