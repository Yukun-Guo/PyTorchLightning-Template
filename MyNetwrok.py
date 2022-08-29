import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.network_modules import Conv2dReLU, Activation
from torchsummary import summary


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super().__init__()
        self.conv2DRelu = Conv2dReLU(in_channels, out_channels, (3, 3), padding=(1, 1), stride=stride,
                                     use_batchnorm=True)
        self.conv2D3x3 = nn.Conv2d(
            out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2D1x1 = nn.Conv2d(
            in_channels, out_channels, (1, 1), stride=stride, bias=False)

    def forward(self, x):
        out = self.conv2DRelu(x)
        out = self.conv2D3x3(out)

        out1 = self.conv2D1x1(x)
        out += out1

        return out


class ResNetBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += x

        return out


class CNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, out_activation):
        super().__init__()

        hid_chns = [64, 96, 128, 192, 128, 96, 64, 32, 16]
        # bottom
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = Conv2dReLU(in_channels, hid_chns[0], kernel_size=(
            7, 7), padding=(3, 3), use_batchnorm=True)  # 1

        # block1
        self.bn2 = nn.BatchNorm2d(hid_chns[0])
        self.relu1 = Activation('relu')
        self.resnet_block1 = ResNetBlock(hid_chns[0], hid_chns[0])  # 1/2
        self.resnet_block2 = ResNetBlock2(hid_chns[0], hid_chns[0])  # 1/2
        self.resnet_block3 = ResNetBlock2(hid_chns[0], hid_chns[0])  # 1/2
        self.bn3 = nn.BatchNorm2d(hid_chns[0])
        self.relu2 = Activation('relu')
        self.pool_msk1 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))

        self.resnet_block4 = ResNetBlock(hid_chns[0], hid_chns[1])  # 1/4
        self.resnet_block5 = ResNetBlock2(hid_chns[1], hid_chns[1])  # 1/4
        self.resnet_block6 = ResNetBlock2(hid_chns[1], hid_chns[1])  # 1/4
        self.resnet_block7 = ResNetBlock2(hid_chns[1], hid_chns[1])  # 1/4
        self.bn4 = nn.BatchNorm2d(hid_chns[1])
        self.relu3 = Activation('relu')
        self.pool_msk2 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))

        self.resnet_block8 = ResNetBlock(hid_chns[1], hid_chns[2])  # 1/8
        self.resnet_block9 = ResNetBlock2(hid_chns[2], hid_chns[2])  # 1/8
        self.resnet_block10 = ResNetBlock2(hid_chns[2], hid_chns[2])  # 1/8
        self.resnet_block11 = ResNetBlock2(hid_chns[2], hid_chns[2])  # 1/8
        self.resnet_block12 = ResNetBlock2(hid_chns[2], hid_chns[2])  # 1/8
        self.resnet_block13 = ResNetBlock2(hid_chns[2], hid_chns[2])  # 1/8
        self.bn5 = nn.BatchNorm2d(hid_chns[2])
        self.relu4 = Activation('relu')
        self.pool_msk3 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))

        self.resnet_block14 = ResNetBlock(hid_chns[2], hid_chns[3])  # 1/16
        self.resnet_block15 = ResNetBlock2(hid_chns[3], hid_chns[3])  # 1/16
        self.resnet_block16 = ResNetBlock2(hid_chns[3], hid_chns[3])  # 1/16
        self.bn6 = nn.BatchNorm2d(hid_chns[3])
        self.relu5 = Activation('relu')
        self.up1 = nn.Upsample(scale_factor=2)  # 1/8
        # cat up1, relu4

        self.conv2 = Conv2dReLU(hid_chns[2] + hid_chns[3], hid_chns[4], kernel_size=(3, 3), padding=(1, 1),
                                use_batchnorm=True)  # 1/8
        self.conv3 = Conv2dReLU(hid_chns[4], hid_chns[4], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1/8
        self.up2 = nn.Upsample(scale_factor=2)
        # cat up2 relu3

        self.conv4 = Conv2dReLU(hid_chns[4] + hid_chns[1], hid_chns[5], kernel_size=(3, 3), padding=(1, 1),
                                use_batchnorm=True)  # 1/4
        self.conv5 = Conv2dReLU(hid_chns[5], hid_chns[5], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1/4
        self.up3 = nn.Upsample(scale_factor=2)
        # cat up3 relu2

        self.conv6 = Conv2dReLU(hid_chns[5] + hid_chns[0], hid_chns[6], kernel_size=(3, 3), padding=(1, 1),
                                use_batchnorm=True)  # 1/2
        self.conv7 = Conv2dReLU(hid_chns[6], hid_chns[6], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1/2
        self.up4 = nn.Upsample(scale_factor=2)
        # cat up2 relu1

        self.conv8 = Conv2dReLU(hid_chns[6] + hid_chns[0], hid_chns[7], kernel_size=(3, 3), padding=(1, 1),
                                use_batchnorm=True)  # 1
        self.conv9 = Conv2dReLU(hid_chns[7], hid_chns[7], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1

        self.conv10 = Conv2dReLU(hid_chns[7], hid_chns[8], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1
        self.conv11 = Conv2dReLU(hid_chns[8], hid_chns[8], kernel_size=(
            3, 3), padding=(1, 1), use_batchnorm=True)  # 1

        self.conv12 = nn.Conv2d(
            hid_chns[8], out_channels, kernel_size=(3, 3), padding=(1, 1))  # 1
        self.out = Activation(out_activation)

    def forward(self, x, mask=None):
        x = self.bn1(x)
        x1 = self.conv1(x)
        if mask is not None:
            x1 = (x1 + 1) * mask
        # block1
        # x = self.pad1(x1)
        # x = self.pool1(x)
        x = self.bn2(x1)
        x = self.relu1(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.bn3(x)
        x2 = self.relu2(x)
        if mask is not None:
            mask = self.pool_msk1(mask)
            x2 = (x2 + 1) * mask

        # block2
        x = self.resnet_block4(x2)
        x = self.resnet_block5(x)
        x = self.resnet_block6(x)
        x = self.resnet_block7(x)
        x = self.bn4(x)
        x3 = self.relu3(x)
        if mask is not None:
            mask = self.pool_msk2(mask)
            x3 = (x3 + 1) * mask

        # block3
        x = self.resnet_block8(x3)
        x = self.resnet_block9(x)
        x = self.resnet_block10(x)
        x = self.resnet_block11(x)
        x = self.resnet_block12(x)
        x = self.resnet_block13(x)
        x = self.bn5(x)
        x4 = self.relu4(x)
        if mask is not None:
            mask = self.pool_msk3(mask)
            x4 = (x4 + 1) * mask

        x = self.resnet_block14(x4)
        x = self.resnet_block15(x)
        x = self.resnet_block16(x)
        x = self.bn6(x)
        x = self.relu5(x)
        x = self.up1(x)
        x5 = torch.cat([x, x4], dim=1)

        x = self.conv2(x5)
        x = self.conv3(x)
        x = self.up2(x)
        # cat up2 relu3
        x6 = torch.cat([x, x3], dim=1)

        x = self.conv4(x6)
        x = self.conv5(x)
        x = self.up3(x)
        # cat up3 relu2
        x7 = torch.cat([x, x2], dim=1)

        x = self.conv6(x7)
        x = self.conv7(x)
        x = self.up4(x)
        # cat up2 relu1
        x8 = torch.cat([x, x1], dim=1)
        x = self.conv8(x8)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.out(x)
        return x

    def __str__():
        return 'CNNNet'


if __name__ == '__main__':
    model = CNNNet(1, 3)
    x = torch.randn(1, 1, 304, 304, requires_grad=True)

    # Export the model
    torch_out = model(x)
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      # where to save the model (can be a file or file-like object)
                      "CNNNet.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    # print model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    summary(model.to(device), (1, 304, 304))
