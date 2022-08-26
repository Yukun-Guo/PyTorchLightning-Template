from typing import Optional
from numpy import pad
import torch
from torch import nn
from torch import functional as F

class Conv2dReLU(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=(1, 1),
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv,bn,relu)
        
class Conv2dReLUSTRPool(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            conv_kernel_size=3,
            padding=1,
            stride=2,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLUSTRPool, self).__init__(conv,bn,relu)

class Conv2dReLUMaxPool(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        pooling = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
        super(Conv2dReLUMaxPool, self).__init__(conv,bn,relu,pooling)


class Conv2dReLUAvgPool(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        pooling = nn.AvgPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
        super(Conv2dReLUAvgPool, self).__init__(conv,bn,relu,pooling)
        

class Conv2dReLUpSample(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor=2,
            mode='nearest',
            align_corners=None,
            kernel_size=3,
            stride=1,
            padding=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        upsample = nn.Upsample(scale_factor=scale_factor,mode=mode,align_corners=align_corners)
        super(Conv2dReLUpSample, self).__init__(conv,bn,relu,upsample)


class Conv2dReLUpTranspose(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor= 2,
            mode='nearest',
            align_corners=None,
            kernel_size=3,
            stride=2,
            padding=1,
            use_batchnorm=True,
    ):
        
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLUpTranspose, self).__init__(conv,bn,relu)


class Conv3dReLU(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=1,
            use_batchnorm=True,
    ):  
        
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv3dReLU, self).__init__(conv,bn,relu)


class Conv3dReLUSTRPool(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            conv_kernel_size=3,
            padding=1,
            stride=2,
            use_batchnorm=True,
    ):  
        
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv3dReLUSTRPool, self).__init__(conv,bn,relu)

class Conv3dReLUMaxPool(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=2,
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            use_batchnorm=True,
    ):  
        
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=stride,padding=padding)
        super(Conv3dReLUMaxPool, self).__init__(conv,bn,relu,pooling)

class Conv3dReLUAvgPool(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=2,
            conv_kernel_size=3,
            conv_stride=1,
            conv_padding=1,
            use_batchnorm=True,
    ):  
        
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        pooling = nn.AvgPool3d(kernel_size=kernel_size, stride=stride,padding=padding)
        super(Conv3dReLUAvgPool, self).__init__(conv,bn,relu,pooling)


class Conv3dReLUUpsample(nn.Sequential):
    
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor=2,
            mode='nearest',
            align_corners=None,
            kernel_size=3,
            stride=1,
            padding=1,
            use_batchnorm=True,
    ):  
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        upsample = nn.Upsample(scale_factor=scale_factor,mode=mode,align_corners=align_corners)
        super(Conv3dReLUUpsample, self).__init__(conv,bn,relu,upsample)


class Conv3dReLUpTranspose(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor= 2,
            mode='nearest',
            align_corners=None,
            kernel_size=3,
            stride=2,
            padding=1,
            use_batchnorm=True,
    ):
        
        conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv3dReLUpTranspose, self).__init__(conv,bn,relu)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'relu':
            self.activation = nn.ReLU()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class ResNetBasicBlock(nn.Module):
    """
       (stride = 2) or in_channels != out_channels           (stride = 1) && in_channels == out_channels

               | —————————————————————                       | ————————————
               ↓                     |                       ↓                |
        conv2d  BN relu              |                  conv2d BN relu        |
               ↓                     ↓                       ↓                |
          conv2d BN relu     conv2d  BN relu          conv2d BN relu          |
               ↓                     ↓                       ↓                |
            conv2d BN                |                    conv2d BN           |
              (+) ———————————————————                       (+) ———————————
               ↓                                             ↓
             relu                                          relu

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1, stride=1, use_batchnorm=True):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding, bias=not use_batchnorm)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride, bias=not use_batchnorm),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        x1 =self.relu(self.bn(self.conv1(x)))
        x2 = self.bn(self.conv2(x1))
        x2 += self.shortcut(x)
        x2 = self.relu(x2)
        return x2

class ResNetBasicBlock3D(nn.Module):
    """
           (stride = 2) or in_channels != out_channels           (stride = 1) && in_channels == out_channels

                   | —————————————————————                       | ————————————
                   ↓                     |                       ↓                |
            conv3d  BN relu              |                  conv3d BN relu        |
                   ↓                     ↓                       ↓                |
              conv3d BN relu     conv3d  BN relu          conv3d BN relu          |
                   ↓                     ↓                       ↓                |
                conv3d BN                |                    conv3d BN           |
                  (+) ———————————————————                       (+) ———————————
                   ↓                                             ↓
                 relu                                          relu

        """
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True):
        super(ResNetBasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not use_batchnorm)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=not use_batchnorm),
                nn.BatchNorm3d(self.expansion * out_channels)
            )

    def forward(self, x):
        x1 = self.relu(self.bn(self.conv1(x)))
        x2 = self.bn(self.conv2(x1))
        x2 += self.shortcut(x)
        out = self.relu(x2)
        return out

def getBaseBlock(name):

    if name is None or name == 'plain2d':
        base_block = Conv2dReLU
    elif name == 'resnet2d':
        base_block = ResNetBasicBlock
    elif name == 'plain3d':
        base_block = Conv3dReLU
    elif name == 'resnet3d':
        base_block = ResNetBasicBlock3D
    elif callable(name):
        base_block = name
    else:
        raise ValueError('BasicBlock should be callable/plain/plain3d/resnet/resnet3d/None; got {}'.format(name))
    return base_block

def getPooling(name):
    if name is None or name == 'maxpool2d':
        pooling = Conv2dReLUMaxPool
    elif name == 'maxpool3d':
        pooling = Conv3dReLUMaxPool
    elif name == 'avgpool2d':
        pooling = Conv2dReLUAvgPool
    elif name == 'avgpool3d':
        pooling = Conv3dReLUAvgPool
    elif name == 'conv2ds2':
        pooling = Conv2dReLUSTRPool
    elif name == 'conv3ds2':
        pooling = Conv3dReLUSTRPool
    elif callable(name):
        pooling = name
    else:
        raise ValueError('BasicBlock should be callable/plain/plain3d/resnet/resnet3d/None; got {}'.format(name))
    return pooling
    
def getUpsample(name):
    if name is None or name == 'upsample2d':
        pooling = Conv2dReLUpSample
    elif name == 'upsample3d':
        pooling = Conv3dReLUUpsample
    elif name == 'transconv2d':
        pooling = Conv2dReLUpTranspose
    elif name == 'transconv3d':
        pooling = Conv3dReLUpTranspose
    elif callable(name):
        pooling = name
    else:
        raise ValueError('BasicBlock should be callable/plain/plain3d/resnet/resnet3d/None; got {}'.format(name))
    return pooling