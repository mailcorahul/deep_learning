import torch.nn as nn


class SSD512(nn.Module):
    """Single Shot MultiBox Detector(SSD) 512x512 Network Implementation"""
    def __init__(self, num_classes=2):

        super().__init__()
        self.input_size = 512
        self.num_classes = num_classes