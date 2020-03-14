import torch
import torch.nn as nn
import torch.nn.functional as F





class Resnet_block(nn.Module):
    def __init__(self, nb_filts, kernels, strides, first=False, downsample=False):
        super(Resnet_block, self).__init__()
        self.first = first
        self.downsample = downsample
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=kernels,
                               padding=(1, 3),
                               stride=strides)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=(1, 3),
                               kernel_size=kernels,
                               stride=1)

        if downsample:


            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(1, 3),
                                             kernel_size=kernels,
                                             stride=strides)
        # self.bn_downsample = nn.BatchNorm2d(num_features = nb_filts[2])

    def forward(self, x):
        identity = x

        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)
        # identity = self.bn_downsample(identity)

        out += identity
        # print(identity.size())
        return out
