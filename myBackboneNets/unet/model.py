import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import base


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        classes: int = 1,
        encoder_channels: List[int] = (16, 32, 64, 128, 256),
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        center_channels: int = 384,
        use_batchnorm: bool = True,
        encoder_blockdepth: Optional[Union[List[int], int]] = 3,
        decoder_blockdepth: Optional[Union[List[int], int]] = 2,
        center_blockdepth: Optional[int] = 2,
        encoder_blocktype: Optional[Union[str, callable]] = 'resnet2d',
        decoder_blocktype: Optional[Union[str, callable]] = 'resnet2d',
        center_blocktype: Optional[Union[str, callable]] = 'resnet2d',
        downsampling_type: Optional[Union[str, callable]] = 'conv2ds2',
        upsampling_type: Optional[Union[str, callable]] = 'transconv2d',
    ):
        super(Unet, self).__init__()

        assert len(encoder_channels) == len(
            decoder_channels), "Encoder channel number must equal to decoder channel number!"

        self.sample_depth = len(encoder_channels)
        self.use_batchnorm = use_batchnorm
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.center_channels = center_channels
        if encoder_blockdepth is not List:
            self.encoder_blockdepth = [encoder_blockdepth, ]*self.sample_depth
        else:
            self.encoder_blockdepth = encoder_blockdepth
        if decoder_blockdepth is not List:
            self.decoder_blockdepth = [decoder_blockdepth, ]*self.sample_depth
        else:
            self.decoder_blockdepth = decoder_blockdepth
        self.center_blockdepth = center_blockdepth
        self.encoder_blocktype = encoder_blocktype
        self.decoder_blocktype = decoder_blocktype
        self.center_blocktype = center_blocktype
        self.downsampling_type = downsampling_type
        self.upsampling_type = upsampling_type

        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.center_block = self._make_center_block()
        self.output = self._make_output()

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_features.append(x)

        x = self.center_block(x)

        for i, decoder in enumerate(self.decoder, 1):
            x = decoder(torch.cat((encoder_features[-i], x), 1))

        x = self.output(x)
        return x

    def _make_encoder(self):
        encoder = []
        encoder_block = base.getBaseBlock(self.encoder_blocktype)
        pooling_layer = base.getPooling(self.downsampling_type)
        for i in range(self.sample_depth):
            if i == 0:
                block1 = encoder_block(
                    self.in_channels, self.encoder_channels[i], kernel_size=3)

                blocks = nn.Sequential(*[block1, *[encoder_block(self.encoder_channels[i],
                                       self.encoder_channels[i], 3) for _ in range(self.encoder_blockdepth[i] - 1)]])

            else:
                pooling = pooling_layer(
                    self.encoder_channels[i - 1], self.encoder_channels[i], kernel_size=2, stride=2)

                blocks = nn.Sequential(*[pooling, *[encoder_block(self.encoder_channels[i],
                                       self.encoder_channels[i], 3) for _ in range(self.encoder_blockdepth[i])]])

            encoder.append(blocks)
        return nn.ModuleList(encoder)

    def _make_decoder(self):
        decoder = []
        decoder_block = base.getBaseBlock(self.decoder_blocktype)
        upsample_layer = base.getUpsample(self.upsampling_type)
        for i in range(self.sample_depth):
            if i == self.sample_depth-1:
                block1 = decoder_block(
                    self.decoder_channels[i]+self.encoder_channels[-(i+1)], self.decoder_channels[i], 3)
                blocks = nn.Sequential(*[block1, *[decoder_block(self.decoder_channels[i],
                                       self.decoder_channels[i], 3) for _ in range(self.decoder_blockdepth[i] - 1)]])

            else:
                block1 = decoder_block(
                    self.decoder_channels[i]+self.encoder_channels[-(i+1)], self.decoder_channels[i], 3)
                upsample = upsample_layer(
                    self.decoder_channels[i], self.decoder_channels[i+1], kernel_size=3)
                blocks = nn.Sequential(*[block1, *[decoder_block(self.decoder_channels[i], self.decoder_channels[i], 3)
                                       for _ in range(self.decoder_blockdepth[i] - 1)], upsample])

            decoder.append(blocks)
        return nn.ModuleList(decoder)

    def _make_center_block(self):
        encoder_block = base.getBaseBlock(self.center_blocktype)
        block_in = encoder_block(
            self.encoder_channels[self.sample_depth - 1], self.center_channels, kernel_size=3)

        block_out = encoder_block(
            self.center_channels, self.decoder_channels[0], kernel_size=3)

        return nn.Sequential(*[block_in, *[encoder_block(self.center_channels, self.center_channels, 3) for _ in range(self.center_blockdepth)], block_out])

    def _make_output(self):
        decoder_block = base.getBaseBlock(self.decoder_blocktype)
        return decoder_block(self.decoder_channels[self.sample_depth - 1], self.classes, 3)


if __name__ == '__main__':

    from torchsummary import summary
    import pytorch_lightning as ptl

    class VisualizeModelStructure(ptl.LightningModule):
        def __init__(self, mymodel, input_size=(1, 512, 512), model_name='model'):
            super(VisualizeModelStructure, self).__init__()
            self.model_name = model_name
            self.input_size = input_size
            self.example_input_array = torch.randn((1, *input_size))
            self.model = mymodel
            self.summary()

        def forward(self, x):
            return self.model(x)

        def summary(self):
            self.to_onnx(file_path=f'{self.model_name}.onnx')
            device = torch.device('cpu')
            if torch.cuda.is_available():
                device = torch.device('cuda')
            summary(self.to(device), self.input_size)

    mymodel = Unet(in_channels=3,
                   classes=3,
                   encoder_channels=[16, 16],
                   decoder_channels=[128, 128],
                   center_channels=368,
                   encoder_blockdepth=3,
                   decoder_blockdepth=2,
                   center_blockdepth=2,
                   encoder_blocktype='resnet2d',
                   decoder_blocktype='plain2d',
                   center_blocktype='resnet2d',
                   downsampling_type='conv2ds2',
                   upsampling_type='transconv2d',
                   )
    VisualizeModelStructure(mymodel, input_size=(3, 512, 512))
