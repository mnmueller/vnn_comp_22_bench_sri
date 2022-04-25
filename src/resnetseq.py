from typing import Optional

import torch
from torch import nn as nn
import torch.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3, in_dim=-1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel

        kernel_size = kernel
        assert kernel_size in [1,2,3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b = []
        layers_b.append(nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=p_1, bias=(not bn)))
        _,_, in_dim = getShapeConv((in_planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=stride, padding=p_1)

        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(nn.Conv2d(planes, self.expansion * planes, kernel_size=kernel_size, stride=1, padding=p_2, bias=(not bn)))
        _,_, in_dim = getShapeConv((planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=1, padding=p_2)
        if bn:
            layers_b.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_b = nn.Sequential(*layers_b)

        layers_a = [torch.nn.Identity()]
        if stride != 1 or in_planes != self.expansion*planes:
            layers_a.append(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=(not bn)))
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_a = nn.Sequential(*layers_a)
        self.out_dim=in_dim

    def forward(self, x):
        out = self.path_a(x) + self.path_b(x)
        return out


def getShapeConv(in_shape, conv_shape, stride = 1, padding = 0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)

class ResNet(nn.Sequential):
    def __init__(self, block, in_ch=3, num_stages=1, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg", in_dim=32, stride=None):
        layers = []
        self.in_planes = in_planes
        if stride is None:
            stride = (num_stages+1) * [2]

        layers.append(nn.Conv2d(in_ch, self.in_planes, kernel_size=3, stride=stride[0], padding=1, bias=not bn))

        _, _, in_dim = getShapeConv((in_ch, in_dim, in_dim), (self.in_planes, 3, 3), stride=stride[0], padding=1)

        if bn:
            layers.append(nn.BatchNorm2d(self.in_planes))

        layers.append(nn.ReLU())

        for s in stride[1:]:
            block_layers, in_dim  = self._make_layer(block, self.in_planes * 2, num_blocks, stride=s, bn=bn, kernel=3, in_dim=in_dim)
            layers.append(block_layers)


        if last_layer == "avg":
            layers.append(nn.AvgPool2d(4))
            layers.append(Flatten())
            layers.append(nn.Linear(self.in_planes * (in_dim//4)**2 * block.expansion, num_classes))
        elif last_layer == "dense":
            layers.append(Flatten())
            layers.append(nn.Linear(self.in_planes * block.expansion * in_dim**2, 100))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(100, num_classes))
        else:
            exit("last_layer type not supported!")

        super(ResNet, self).__init__(*layers)

    def _make_layer(self, block, planes, num_layers, stride, bn, kernel, in_dim=None):
        strides = [stride] + [1]*(num_layers-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel, in_dim=in_dim))
            in_dim = layers[-1].out_dim
            layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), in_dim

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def resnet2b(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=1, num_blocks=2, in_planes=8, bn=bn, last_layer="dense")

def resnet2b2(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=2, num_blocks=1, in_planes=16, bn=bn, last_layer="dense", stride=[2,2,2])

def resnet2b3(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=2, num_blocks=1, in_planes=32, bn=bn, last_layer="dense", stride=[1,2,2])

def resnet2b4(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=2, num_blocks=1, in_planes=12, bn=bn, last_layer="dense", stride=[1,2,2])

def resnet3b2(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=3, num_blocks=1, in_planes=16, bn=bn, last_layer="dense", stride=[2,2,2,2])

def resnet4b(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=2, num_blocks=2, in_planes=8, bn=bn, last_layer="avg")

def resnet4b2(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=4, num_blocks=1, in_planes=16, bn=bn, last_layer="dense", stride=[2,2,2,1,1])

def resnet4b1(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=4, num_blocks=1, in_planes=16, bn=bn, last_layer="dense", stride=[1,1,2,2,2])

def resnet9b(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=3, num_blocks=3, in_planes=16, bn=bn, last_layer="dense")

def resnet8b(bn=False, in_ch=3, in_dim=32):
    return ResNet(BasicBlock, in_ch=in_ch, num_stages=4, num_blocks=2, in_planes=16, bn=bn, last_layer="dense")

Models = {
    'resnet4b': resnet4b,
    'resnet4b1': resnet4b1,
    'resnet4b2': resnet4b2,
    'resnet3b2': resnet3b2,
    'resnet2b2': resnet2b2,
    'resnet2b3': resnet2b3,
    'resnet2b4': resnet2b4,
    'resnet2b': resnet2b,
    'resnet9b': resnet9b,
    'resnet8b': resnet8b,
}