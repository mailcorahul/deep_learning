import torch
import torch.nn as nn
import torchvision.models as models

class ConvTransposeBlock(nn.Module):
    """ A ConvTranspose-Relu-BatchNorm block """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv_transpose = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_transpose(x)

class DeFocusNet(nn.Module):
    """A simple Defocus estimation Network.
    Uses resnet50 backbone and some set of conv transpose blocks later on to predict defocus map.
    """
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.deconv0 = ConvTransposeBlock(in_channels=2048, out_channels=1024, kernel_size=2, stride=2)
        self.deconv1 = ConvTransposeBlock(in_channels=1024, out_channels=512, kernel_size=4, padding=1,stride=2)
        self.deconv2 = ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=6, padding=2, stride=2)
        self.deconv3 = ConvTransposeBlock(in_channels=256, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.deconv4 = ConvTransposeBlock(in_channels=64, out_channels=1, kernel_size=2, stride=2)
        self.estimator = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)


    def forward(self, input):

        feature_volume = self.backbone(input)
        #print(feature_volume.size())
        level0 = self.deconv0(feature_volume)
        #print(level0.size())
        level1 = self.deconv1(level0)
        #print(level1.size())
        level2 = self.deconv2(level1)
        #print(level2.size())
        level3 = self.deconv3(level2)
        #print(level3.size())
        level4 = self.deconv4(level3)
        #print(level4.size())

        output_map = self.estimator(level4)
        #print(output_map.size())
        return output_map