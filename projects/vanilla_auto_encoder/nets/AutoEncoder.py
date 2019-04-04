import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """A simple undercomplete AutoEncoder"""
    def __init__(self):        
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Linear(784, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU()                        
                        )
        self.decoder = nn.Sequential(
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 784),
                        nn.Sigmoid()
                        )

    def forward(self, x):

        x_encoded = self.encoder(x)
        y = self.decoder(x_encoded)
        return y