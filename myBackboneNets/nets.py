import segmentation_models_pytorch as smp
import pytorch_lightning as ptl
import torch
from torchsummary import summary


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


model = smp.Unet(encoder_name='mit_b0', encoder_weights=None,
                 in_channels=3, classes=12)
print(model)
VisualizeModelStructure(model, input_size=(3, 512, 512))
