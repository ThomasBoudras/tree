import torch.nn as nn

import math

import torch
import torch.nn as nn
import numpy as np 

from src.utils import utils

log = utils.get_logger(__name__)


### Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(
            self,
            n_resblocks,
            scale,
            n_feats,
            n_channels,
            pretrained_model_path,
            n_resgroups,
            reduction,
            ):
        super(RCAN, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        self.n_channels = n_channels

        conv=default_conv

        # define head module
        modules_head = [conv(n_channels, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_channels, kernel_size)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        
        if pretrained_model_path :
            self.load_partial_weight(pretrained_model_path)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 
            
    def load_partial_weight(self, pretrained_model) :
        log.info(f"Using the pre-trained model {pretrained_model} to initialise the model")
        load_from = torch.load(pretrained_model,  map_location=torch.device('cpu'), weights_only=True)

        for module_name , module_tensor in load_from.items():
            if module_name == "head.0.weight"  :
                load_from[module_name] = module_tensor.repeat(1, int(np.ceil(self.n_channels/3)), 1, 1)[:,:self.n_channels, :,:]
            
            if module_name == "tail.1.weight" :
                load_from[module_name] = module_tensor.repeat(int(np.ceil(self.n_channels/3)), 1, 1, 1)[:self.n_channels, :, :, :]

            if module_name == "tail.1.bias" :
                load_from[module_name] = module_tensor.repeat(int(np.ceil(self.n_channels/3)))[:self.n_channels]


        self.load_state_dict(load_from, strict=False)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
