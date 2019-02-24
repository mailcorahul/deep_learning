import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Flattens the final conv layer output for FC to take in.
    Args:
        x: input tensor of size [256, c, h, w]
    Returns:
        x: flattened input tensor of size [256, c*h*w]
    """
    
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Net(nn.Module):
    """ Simple Convolutional Network with n layers"""

    def __init__(self):
        """
        Creates a sequential container of conv, relu, linear and softmax modules.
        """

        super(Net, self).__init__();
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32768, 10),
            nn.Softmax()
            );

    def forward(self, x):
        """
        Forward pass
        """

        y = self.conv(x);
        return y;