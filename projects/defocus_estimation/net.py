import torch
import torch.nn as nn
import torchvision.models as models

class DeFocusNet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-3])
        self.estimator = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)

    def forward(self, input):

        feature_volume = self.backbone(input)
        output_map = self.estimator(feature_volume)

        return output_map