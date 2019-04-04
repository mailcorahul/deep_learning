import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """A simple undercomplete AutoEncoder using ConvNets"""
    def __init__(self):        
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 32, 3),
                        nn.Relu(),
                        nn.Conv2d(32, 64, 3),
                        nn.Relu()                        
                        )
        self.decoder = nn.Sequential(
                        nn.Conv2d(64, 32, 3),
                        nn.Relu(),
                        nn.Conv2d(32, 1, 3),
                        )        

    def forward(self, x):

        x_encoded = self.encoder(x)
        y = self.decoder(x_encoded)
        return y