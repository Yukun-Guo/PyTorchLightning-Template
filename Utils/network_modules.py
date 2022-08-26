import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary


class Conv2dReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0), stride=(1, 1), use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Conv3dReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0, 0), stride=(1, 1, 1), use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=not use_batchnorm)

        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv3dReLU, self).__init__(conv, bn, relu)


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
            raise ValueError(
                'Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


'''ResNet'''


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

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(
            3, 3), stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(
            3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1(x)))
        x2 = self.bn(self.conv2(x1))
        x2 += self.shortcut(x)
        x2 = F.relu(x2)
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

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * out_channels)
            )

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1(x)))
        x2 = self.bn(self.conv2(x1))
        x2 += self.shortcut(x)
        return F.relu(x2)


'''Inception'''


class InceptionV4A(nn.Module):
    #                     Concat Output
    #                        ↑
    #    +------------------------------------+
    #    ↑           ↑            ↑           ↑
    #    |           |            |       3x3 Conv
    #    |           |            |           ↑
    #    |        1x1 Conv     5x5 Conv   3x3 Conv
    #    |                         ↑          ↑
    # 1x1 Conv   Avg Pooling   1x1 Conv   1x1 Conv
    #    ↑            ↑            ↑          ↑
    #    +------------------------------------+
    #                        ↑
    #                      input

    def __init__(self, in_channels, out_channels):
        super(InceptionV4A, self).__init__()
        self.out_chn = int(out_channels / 4)
        self.out_channels = out_channels
        self.conv1_1 = nn.Conv2d(
            in_channels, self.out_chn, kernel_size=(1, 1), padding=0)
        self.conv1_2 = nn.Conv2d(
            self.out_chn, self.out_chn, kernel_size=(3, 3), padding=1)
        self.conv1_3 = nn.Conv2d(
            self.out_chn, self.out_chn, kernel_size=(3, 3), padding=1)

        self.conv2_1 = nn.Conv2d(
            in_channels, self.out_chn, kernel_size=(1, 1), padding=0)
        self.conv2_2 = nn.Conv2d(
            self.out_chn, self.out_chn, kernel_size=(5, 5), padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels, self.out_chn, kernel_size=(1, 1), padding=0)

        self.conv4_1 = nn.Conv2d(
            in_channels, self.out_chn, kernel_size=(1, 1), padding=0)

        self.bn = nn.BatchNorm2d(self.out_chn)
        self.outconv = nn.Conv2d(
            in_channels=self.out_chn * 4, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1_1(x)))
        x1 = F.relu(self.bn(self.conv1_2(x1)))
        x1 = F.relu(self.bn(self.conv1_3(x1)))

        x2 = F.relu(self.bn(self.conv2_1(x)))
        x2 = F.relu(self.bn(self.conv2_2(x2)))

        x3 = self.avgpool(x)
        x3 = F.relu(self.bn(self.conv3_2(x3)))

        x4 = F.relu(self.bn(self.conv4_1(x)))

        out = torch.cat([x1, x2, x3, x4], 1)
        if self.out_chn * 4 != self.out_channels:
            out = self.outconv(out)
        return out


class InceptionV4A3D(nn.Module):
    #                     Concat Output
    #                        ↑
    #    +------------------------------------+
    #    ↑           ↑            ↑           ↑
    #    |           |            |       3x3 Conv
    #    |           |            |           ↑
    #    |        1x1 Conv     3x3 Conv   3x3 Conv
    #    |                         ↑          ↑
    # 1x1 Conv   Avg Pooling   1x1 Conv   1x1 Conv
    #    ↑            ↑            ↑          ↑
    #    +------------------------------------+
    #                        ↑
    #                      input

    def __init__(self, in_channels, out_channels):
        super(InceptionV4A3D, self).__init__()
        self.out_chn = int(out_channels / 4)
        self.out_channels = out_channels
        self.conv1_1 = nn.Conv3d(
            in_channels, self.out_chn, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv3d(
            self.out_chn, self.out_chn, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv3d(
            self.out_chn, self.out_chn, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv3d(
            in_channels, self.out_chn, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv3d(
            self.out_chn, self.out_chn, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv3d(
            in_channels, self.out_chn, kernel_size=1, padding=0)
        self.conv4_1 = nn.Conv3d(
            in_channels, self.out_chn, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm3d(self.out_chn)
        self.outconv = nn.Conv3d(
            in_channels=self.out_chn * 4, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1_1(x)))
        x1 = F.relu(self.bn(self.conv1_2(x1)))
        x1 = F.relu(self.bn(self.conv1_3(x1)))

        x2 = F.relu(self.bn(self.conv2_1(x)))
        x2 = F.relu(self.bn(self.conv2_2(x2)))

        x3 = self.avgpool(x)
        x3 = F.relu(self.bn(self.conv3_2(x3)))

        x4 = F.relu(self.bn(self.conv4_1(x)))

        out = torch.cat([x1, x2, x3, x4], 1)
        if self.out_chn * 4 != self.out_channels:
            out = self.outconv(out)
        return out


'''DensNet'''


class DensNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(DensNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class DensNetBasicBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(DensNetBasicBlock3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class DensNetBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(DensNetBottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DensNetBottleneckBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(DensNetBottleneckBlock3D, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_planes)
        self.conv2 = nn.Conv3d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class DensNetTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, pooling=False):
        super(DensNetTransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.pooling = pooling

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        if self.pooling:
            out = F.avg_pool2d(out, 2)
        return out


class DensNetTransitionBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, pooling=False):
        super(DensNetTransitionBlock3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.pooling = pooling

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        if self.pooling:
            out = F.avg_pool3d(out, 2)
        return out


class DensNetDenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DensNetDenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = [block(in_planes + i * growth_rate, growth_rate, dropRate)
                  for i in range(nb_layers)]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


'''example'''


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.inblock = InceptionV4A(3, 8)
        self.encoder1 = ResNetBasicBlock(32, 64, 2)
        self.encoder2 = ResNetBasicBlock(64, 128, 2)
        self.encoder3 = ResNetBasicBlock(128, 256, 2)
        self.encoder4 = ResNetBasicBlock(256, 512, 2)
        self.mid = ResNetBasicBlock(512, 512, 1)
        self.decoder4 = ResNetBasicBlock(512, 256, 1)
        self.decoder3 = ResNetBasicBlock(512, 128, 1)
        self.decoder2 = ResNetBasicBlock(256, 64, 1)
        self.decoder1 = ResNetBasicBlock(128, 32, 1)
        self.outblock = ResNetBasicBlock(64, 16, 1)
        self.out = nn.Conv2d(16, 2, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(2)
        self.tconv4 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv1 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        inb = self.inblock(x)
        ecd1 = self.encoder1(inb)
        ecd2 = self.encoder2(ecd1)
        ecd3 = self.encoder3(ecd2)
        ecd4 = self.encoder4(ecd3)

        md = self.mid(ecd4)

        dcd4 = self.decoder4(md)
        dcd3 = self.decoder3(torch.cat([F.relu(self.tconv4(dcd4)), ecd3], 1))
        dcd2 = self.decoder2(torch.cat([F.relu(self.tconv3(dcd3)), ecd2], 1))
        dcd1 = self.decoder1(torch.cat([F.relu(self.tconv2(dcd2)), ecd1], 1))
        outb = F.relu(self.outblock(
            torch.cat([F.relu(self.tconv1(dcd1)), inb], 1)))
        out = F.relu(self.bn(self.out(outb)))
        return out


class DenseNet(nn.Module):
    def __init__(self, input_channel, depth, num_classes, growth_rate=12, reduction=0.5, bottleneck=False, dropRate=0.0):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        block = DensNetBottleneckBlock if bottleneck == True else DensNetBasicBlock
        n = depth
        self.conv1 = nn.Conv2d(input_channel, in_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = DensNetTransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)

        in_planes = int(math.floor(in_planes * reduction))
        self.block2 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = DensNetTransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)

        in_planes = int(math.floor(in_planes * reduction))
        self.block3 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans3 = DensNetTransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)

        in_planes = int(math.floor(in_planes * reduction))
        self.block4 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.outconv = nn.Conv2d(
            in_planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.trans3(self.block3(out))
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.outconv(out))
        return out


class DenseNet3D(nn.Module):
    def __init__(self, input_channel, depth, num_classes, growth_rate=12, reduction=0.5, bottleneck=False, dropRate=0.0):
        super(DenseNet3D, self).__init__()
        in_planes = 2 * growth_rate
        block = DensNetBottleneckBlock3D if bottleneck else DensNetBasicBlock3D
        n = depth
        self.conv1 = nn.Conv3d(input_channel, in_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = DensNetTransitionBlock3D(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)

        in_planes = int(math.floor(in_planes * reduction))
        self.block2 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = DensNetTransitionBlock3D(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)

        in_planes = int(math.floor(in_planes * reduction))
        self.block4 = DensNetDenseBlock(
            n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.outconv = nn.Conv3d(
            in_planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        # out = self.trans3(self.block3(out))
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.outconv(out))
        return out


if __name__ == '__main__':
    writer = SummaryWriter('logs')

    #
    # net = ExampleCNN()
    net = DenseNet3D(1, depth=4, num_classes=8, growth_rate=15).to('cuda:0')
    summary(net, torch.randn(1, 1, 128, 128, 128).cuda())
    out = net(torch.randn(1, 1,  128, 128, 128).cuda())
    # writer.add_graph(net, torch.randn(1, 3, 224, 224))
    # writer.close()
    torch.save(net, 'test.pt')
