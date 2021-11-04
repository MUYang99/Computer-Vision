from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = [nn.Conv2d(3, down_filter_sizes[0], kernel_sizes[0], padding=conv_paddings[0]),
                            nn.Conv2d(down_filter_sizes[0], down_filter_sizes[1], kernel_sizes[1], padding=conv_paddings[1]),
                            nn.Conv2d(down_filter_sizes[1], down_filter_sizes[2], kernel_sizes[2], padding=conv_paddings[2]),
                            nn.Conv2d(down_filter_sizes[2], down_filter_sizes[3], kernel_sizes[3], padding=conv_paddings[3])]
        layers_bn_down = [nn.BatchNorm2d(down_filter_sizes[0]),
                          nn.BatchNorm2d(down_filter_sizes[1]),
                          nn.BatchNorm2d(down_filter_sizes[2]),
                          nn.BatchNorm2d(down_filter_sizes[3])]
        layers_pooling = [nn.MaxPool2d(pooling_kernel_sizes[0], pooling_strides[0], return_indices=True),
                          nn.MaxPool2d(pooling_kernel_sizes[1], pooling_strides[1], return_indices=True),
                          nn.MaxPool2d(pooling_kernel_sizes[2], pooling_strides[2], return_indices=True),
                          nn.MaxPool2d(pooling_kernel_sizes[3], pooling_strides[3], return_indices=True)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = [nn.Conv2d(down_filter_sizes[3], up_filter_sizes[0], kernel_sizes[0], stride=2, padding=conv_paddings[0]),
                          nn.Conv2d(up_filter_sizes[0], up_filter_sizes[1], kernel_sizes[1], stride=2, padding=conv_paddings[1]),
                          nn.Conv2d(up_filter_sizes[1], up_filter_sizes[2], kernel_sizes[2], stride=2, padding=conv_paddings[2]),
                          nn.Conv2d(up_filter_sizes[2], up_filter_sizes[3], kernel_sizes[3], stride=2, padding=conv_paddings[3])]
        layers_bn_up = [nn.BatchNorm2d(up_filter_sizes[0]),
                        nn.BatchNorm2d(up_filter_sizes[1]),
                        nn.BatchNorm2d(up_filter_sizes[2]),
                        nn.BatchNorm2d(up_filter_sizes[3])]
        layers_unpooling = [nn.MaxUnpool2d(pooling_kernel_sizes[0], pooling_strides[0]),
                            nn.MaxUnpool2d(pooling_kernel_sizes[1], pooling_strides[1]),
                            nn.MaxUnpool2d(pooling_kernel_sizes[2], pooling_strides[2]),
                            nn.MaxUnpool2d(pooling_kernel_sizes[3], pooling_strides[3])]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.final_conv = nn.Conv2d(up_filter_sizes[3], 11, kernel_size=1)

    def forward(self, x):
        x = self.layers_conv_down[0](x) # 3 x 64 x 64 -> 32 x 64 x 64
        x = self.layers_bn_down[0](x)
        x = self.relu(x)
        x = self.layers_pooling[0](x) # 32 x 64 x 64 -> 32 x 32 x 32

        x = self.layers_conv_down[1](x) # 32 x 32 x 32 -> 64 x 32 x 32
        x = self.layers_bn_down[1](x)
        x = self.relu(x)
        x = self.layers_pooling[1](x) # 64 x 32 x 32 -> 64 x 16 x 16

        x = self.layers_conv_down[2](x)  # 64 x 16 x 16 -> 128 x 16 x 16
        x = self.layers_bn_down[2](x)
        x = self.relu(x)
        x = self.layers_pooling[2](x)  # 128 x 16 x 16 -> 128 x 8 x 8

        x = self.layers_conv_down[3](x)  # 128 x 8 x 8 -> 256 x 8 x 8
        x = self.layers_bn_down[3](x)
        x = self.relu(x)
        x = self.layers_pooling[3](x)  # 256 x 8 x 8 -> 256 x 4 x 4

        x = self.layers_unpooling[0](x) # 256 x 4 x 4 -> 256 x 8 x 8
        x = self.layers_conv_up[0](x) # 256 x 8 x 8 -> 128 x 8 x 8
        x = self.layers_bn_up[0](x)
        x = self.relu(x)

        x = self.layers_unpooling[1](x)
        x = self.layers_conv_up[1](x)
        x = self.layers_bn_up[1](x)
        x = self.relu(x)

        x = self.layers_unpooling[2](x)
        x = self.layers_conv_up[2](x)
        x = self.layers_bn_up[2](x)
        x = self.relu(x)

        x = self.layers_unpooling[3](x)
        x = self.layers_conv_up[3](x)
        x = self.layers_bn_up[3](x)
        x = self.relu(x)

        x = self.final_conv(x)


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
