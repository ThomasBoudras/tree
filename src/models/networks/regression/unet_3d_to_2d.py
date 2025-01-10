import torch
import torch.nn as nn
import torch.nn.functional as F


# Parts of the U-Net model
# Inspired from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class DoubleConv3D(nn.Module):
    """Double convolution 3D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv (3D version)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),  # Max pooling 3D
            DoubleConv3D(in_channels, out_channels)  # Double convolution 3D
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def maxpool_3d_to_2d(input_tensor):
    output_tensor,  = torch.max(input_tensor, dim=2)
    return output_tensor

def averagepool_3d_to_2d(input_tensor):
    output_tensor = torch.mean(input_tensor, dim=2)
    return output_tensor


# Full assembly of the parts to form the complete network
class UNet(nn.Module):
    def __init__(
            self,
            n_channels_in,
            time_steps,
            bilinear=False,
            out_activation=None,
            function_3d_to_2d = maxpool_3d_to_2d
            ):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels_in
        self.bilinear = bilinear
        self.inc = DoubleConv3D(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1 
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.out_activation = None
        if out_activation is not None:
            if (out_activation == "None") or (out_activation is None) or (out_activation == "null"):
                self.out_activation = None
            else:
                self.out_activation = out_activation
        self.function_3d_to_2d = function_3d_to_2d
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1 = self.function_3d_to_2d(x1)
        x2 = self.function_3d_to_2d(x2)
        x3 = self.function_3d_to_2d(x3)
        x4 = self.function_3d_to_2d(x4)
        x5 = self.function_3d_to_2d(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
                                  
        if self.out_activation is not None:
            output = self.out_activation(output)  # eg relu to avoid negative predictions
        # Now the output will have the same WxH as the input
        return output

