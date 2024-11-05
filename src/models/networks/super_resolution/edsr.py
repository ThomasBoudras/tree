import torch.nn as nn

import math

import torch
import torch.nn as nn

from src.utils import utils

log = utils.get_logger(__name__)
import numpy as np

class EDSR(nn.Module):
    def __init__(
            self,
            n_resblocks,
            scale,
            n_feats,
            n_channels,
            res_scale,
            pretrained_model_path,
            ):
        
        super(EDSR, self).__init__()
        conv=default_conv
        kernel_size = 3 
        act = nn.ReLU(True)

        self.n_channels = n_channels

        # define head module
        m_head = [conv(n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_channels, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

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
            nb_repeat = int(np.ceil(self.n_channels/3))
            if module_name == "head.0.weight"  :
                load_from[module_name] = module_tensor.repeat(1, nb_repeat, 1, 1)[:,:self.n_channels, :,:]
            
            if module_name == "tail.1.weight" :
                load_from[module_name] = module_tensor.repeat(nb_repeat, 1, 1, 1)[:self.n_channels, :, :, :]

            if module_name == "tail.1.bias" :
                load_from[module_name] = module_tensor.repeat(nb_repeat)[:self.n_channels]


        self.load_state_dict(load_from, strict=False)




def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


