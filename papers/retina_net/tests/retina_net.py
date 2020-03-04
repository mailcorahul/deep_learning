## Script for testing/debugging RetinaNet network

## 1. Initialize network
## 2. Pass input image and check output shape
## 3. Check for output class and box ranges
## 4. Visualize anchor boxes

import sys
sys.path.append('..')

import torch

from nets.fpn import FPN

if __name__ == '__main__':

    # 1. initialize backbone
    num_classes = 2
    fpn = FPN(backbone_name='resnet50', use_pretrained=True, num_classes=num_classes)
    print(fpn)

    # 2. forward pass with a random input
    input_image = torch.zeros((1, 3, 500, 500)).random_()
    print('[/] shape of input batch: {}'.format(input_image.size()))
    output = fpn(input_image)
