import numpy as np

import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def forward(self, y):
        (batch, ch, h, w) = y.size()
        features = y.view(batch, ch, w * h)
        features_t = features.transpose(1, 2)
        gram_matrix = features.bmm(features_t) / (ch * h * w)
        return gram_matrix


class Inspiration(nn.Module):

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        # буфер без параметров
        self.G = torch.Tensor(B, C, C)
        self.C = C
        self.reset_param()

    def setTarget(self, target):
        self.G = target

# заполним тензор числами из непрерывного равномерного распределения.
    def reset_param(self):
        self.weight.data.uniform_(0.0, 0.02)


# х- в качестве карты признаков 3D.
    def forward(self, X):
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(
            self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
            X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x ' + str(self.C) + ')'


class ConvLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflect_padding = int(np.floor(kernel_size / 2))
        self.reflect_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

    def forward(self, x):
        out = self.reflect_pad(x)
        out = self.conv2d(out)
        return out


class UpsampConvLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride,
                 upsample=None):
        super(UpsampConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        self.reflect_padding = int(np.floor(kernel_size / 2))
        if self.reflect_padding != 0:
            self.reflect_pad = nn.ReflectionPad2d(self.reflect_padding)
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflect_padding != 0:
            x = self.reflect_pad(x)
        out = self.conv2d(x)
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsamp=None,
                 norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsamp = downsamp
        if self.downsamp is not None:
            self.residual_layer = nn.Conv2d(in_planes, planes * self.expansion,
                                            kernel_size=1, stride=stride)
        conv_block = []
        conv_block += [norm_layer(in_planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       ConvLayer(planes, planes, kernel_size=3, stride=stride)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                 stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        if self.downsamp is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(nn.Module):
    """ Up-sample residual block.
    """

    def __init__(self, in_planes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampConvLayer(in_planes,
                                                planes * self.expansion,
                                                kernel_size=1, stride=1,
                                                upsample=stride)
        conv_block = []
        conv_block += [norm_layer(in_planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       UpsampConvLayer(planes, planes, kernel_size=3,
                                         stride=1, upsample=stride)]
        conv_block += [norm_layer(planes, track_running_stats=True),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                 stride=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.residual_layer(x) + self.conv_block(x)


class MsgNet(nn.Module):


    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=[]):
        super(MsgNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = []
        model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                   norm_layer(64, track_running_stats=True),
                   nn.ReLU(inplace=True),
                   block(64, 32, 2, 1, norm_layer),
                   block(32 * expansion, ngf, 2, 1, norm_layer)]
        self.model1 = nn.Sequential(*model1)

        model = []
        self.ins = Inspiration(ngf * expansion)
        model += [self.model1]
        model += [self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(16 * expansion, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def setTarget(self, Xs):
        F = self.model1(Xs)
        G = self.gram(F)
        self.ins.setTarget(G)

    def forward(self, input):
        return self.model(input)
