import torch.nn as nn
import torchvision.models as models

class FPN(nn.Module):
    """Implementation of Feature Pyramid Network"""

    def __init__(self, backbone_name='resnet50', use_pretrained=True, num_classes=-1):
        super().__init__()

        # fpn parameters
        self.num_classes = num_classes
        self.anchors = 1
        self.num_feature_levels = 4
        self.features = []
        self.channels = [256, 512, 1024, 2048]
        self.filters = [3, 5, 7, 9]
        self.dims_reduction = []

        # initialize backbone
        self.backbone_name = backbone_name
        self.backbone = getattr(models, self.backbone_name)(pretrained=use_pretrained)

        try:
            # add hook for multiple feature map levels
            for i in range(1, self.num_feature_levels+1):
                self.backbone._modules['layer%d' % (i)][-1].register_forward_hook(self.get_feature_map)

            # initialize classifier modules for multiple feature levels
            self.feature_classifiers = []
            for i in range(self.num_feature_levels):
                self.feature_classifiers.append(
                    nn.Conv2d(self.channels[i], self.num_classes * self.anchors, self.filters[i], stride=1)
                )

            self.dims_reduction = [None] * self.num_feature_levels
            for i in range(self.num_feature_levels - 1):
                self.dims_reduction[i] = nn.Conv2d(in_channels=self.channels[i+1],                                                  out_channels=self.channels[i], kernel_size=1)

            self.conv3x3 = []
            for i in range(self.num_feature_levels):
                self.conv3x3.append(nn.Conv2d(in_channels=self.channels[i], out_channels=self.channels[i], kernel_size=3, padding=1))

            self.conv1x1 = nn.Conv2d(in_channels=self.channels[-1], out_channels=self.channels[-1], kernel_size=1)

        except Exception as e:
            raise Exception("error while initialising FPN network: {}".format(e))

    def get_feature_map(self, module, input, output):
        self.features.append(output)

    def forward(self, x):
        # forward propagate the input batch
        self.backbone(x)
        print('\n[/] shapes of multi-level features')
        for i in range(self.num_feature_levels):
            print(self.features[i].size())

        # create pyramid features
        print('\n[/] shapes of pyramidal features')
        pyramid_features = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels - 2, -1, -1):

            # upsample the previous feature map
            upsampling_size = (self.features[i].size(-1), self.features[i].size(-2))
            upsampled_i1 = nn.UpsamplingNearest2d(size=upsampling_size)(self.features[i+1])

            # reduce channels using 1x1 conv
            feature_i1 = self.dims_reduction[i](upsampled_i1)

            # element-wise addition
            merged_feature_map = feature_i1 + self.features[i]

            # 3x3 conv to generate final feature map
            pyramid_features[i] = self.conv3x3[i](merged_feature_map)

            print(pyramid_features[i].size())

        # for top-most level
        pyramid_features[-1] = self.conv1x1(self.features[-1])

        print('\n[/] shapes of output feature maps')
        for i in range(self.num_feature_levels):
            output_i = self.feature_classifiers[i](pyramid_features[i])
            print(output_i.size())
