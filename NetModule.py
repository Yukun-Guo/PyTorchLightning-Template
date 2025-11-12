import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import functional as FM
from lightning.pytorch.callbacks import early_stopping, model_checkpoint, lr_monitor
from lightning.pytorch.loggers import TensorBoardLogger
from Network import CNNNet
from losses import dice
from torchsummary import summary
from Network import CNNNet

class NetModule(L.LightningModule):

    def __init__(self,
                 config):
        super(NetModule, self).__init__()

        self.save_hyperparameters()
        
        self.input_size = config['DataModule']['image_shape'][:2]
        self.img_chn = config['DataModule']['image_shape'][2]
        self.n_class = config['DataModule']['n_class']
        self.example_input_array = torch.randn((1, self.img_chn, *self.input_size))
        self.out = CNNNet(in_channels=self.img_chn, out_channels=self.n_class,out_activation=None)

        self.model_name = config['NetModule']["model_name"]
        self.log_dir = config['NetModule']["log_dir"]
        self.k_fold = config['DataModule']["k_fold"]
        self.valid_dataset = None
        self.train_dataset = None
    
    def forward(self, x):
        return self.out(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat_s = F.softmax(y_hat, dim=1, _stacklevel=5)
        train_loss = F.cross_entropy(y_hat, y) + dice.DiceLoss(mode='multiclass')(y_hat_s, y)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat_s = F.softmax(y_hat, dim=1, _stacklevel=5)
        val_loss = F.cross_entropy(y_hat_s, y) + dice.DiceLoss(mode='multiclass')(y_hat_s, y)
        val_iou = FM.jaccard_index(y_hat_s, y, task='multiclass', num_classes=self.n_class)
        self.log_dict({'val_loss': val_loss, 'val_iou': val_iou},prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)  # 0
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-8)
        lr_scheduler = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'reduce_on_plateau': True
        }
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        fd = str(self.k_fold)
        early_stop = early_stopping.EarlyStopping(monitor="val_loss",
                                                  min_delta=1e-08,
                                                  patience=10)

        checkpoint = model_checkpoint.ModelCheckpoint(
            dirpath=self.log_dir + self.model_name,
            monitor="val_loss",
            save_top_k=1,
            verbose=True,
            filename=f'{self.model_name}-fold={fd}' +
            '-{epoch:03d}-{val_loss:.5f}')

        lr_monitors = lr_monitor.LearningRateMonitor(logging_interval='epoch')
        return [early_stop, checkpoint, lr_monitors]

    def configure_loggers(self):
        return TensorBoardLogger(self.log_dir, name=self.model_name)

    def summary(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        summary(self.to(device), (1, 480, 288))


if __name__ == '__main__':

    model = NetModule(CNNNet)
    model.summary()
    model.to_onnx('test.onnx')
