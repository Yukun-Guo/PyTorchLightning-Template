import torch
import torch.nn.functional as F
import pytorch_lightning as ptl
from torchmetrics import functional as FM
from pytorch_lightning.callbacks import early_stopping, model_checkpoint, lr_monitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from MyNetwrok import CNNNet
from losses import dice, soft_bce, soft_ce
from Utils.utils import apply_colormap
from torchsummary import summary
from torch.utils.data import DataLoader
from PreprocessData import myDataset_mat
from Utils import resize_right


class NetModel(ptl.LightningModule):

    def __init__(self,
                 backbone_net,
                 input_size=(384, 288),
                 img_chn=1,
                 log_dir='logs/',
                 k_fold=0):
        super(NetModel, self).__init__()

        self.save_hyperparameters()
        self.input_size = input_size
        self.img_chn = img_chn
        self.example_input_array = torch.randn((1, img_chn, *input_size))
        self.out1 = backbone_net(in_channels=1,
                                 out_channels=1,
                                 out_activation=None)
        self.out2 = backbone_net(in_channels=1,
                                 out_channels=12,
                                 out_activation=None)
        self.model_name = backbone_net.__str__()
        self.log_dir = log_dir
        self.valid_dataset = None
        self.train_dataset = None
        self.k_fold = k_fold

    def forward(self, x):
        x1 = self.out1(x)
        x2 = self.out2(x, x1)

        return x1, x2

    def training_step(self, batch, batch_idx):
        x, (y1, y2) = batch
        y_hat1, y_hat2 = self.forward(x)
        y_hat1_s = torch.squeeze(torch.sigmoid(y_hat1), 1)
        y_hat2_s = F.softmax(y_hat2, dim=1, _stacklevel=5)

        loss1 = soft_bce.SoftBCEWithLogitsLoss()(
            y_hat1_s, y1) + dice.DiceLoss(mode='binary')(y_hat1_s, y1)
        loss2 = F.cross_entropy(y_hat2, y2) + \
            dice.DiceLoss(mode='multiclass')(y_hat2_s, y2)

        train_loss = (loss1 + loss2) / 2

        iou1 = FM.jaccard_index(y_hat1_s, y1, task='binary')
        iou2 = FM.jaccard_index(
            y_hat2_s, y2, task='multiclass', num_classes=12)
        train_iou = (iou1 + iou2) / 2

        self.logger.experiment.add_scalars("losses",
                                           {"train_loss": train_loss},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars("iou", {'train_iou': train_iou},
                                           global_step=self.global_step)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, (y1, y2) = batch
        y_hat1, y_hat2 = self.forward(x)
        y_hat1_s = torch.squeeze(torch.sigmoid(y_hat1), 1)
        y_hat2_s = F.softmax(y_hat2, dim=1, _stacklevel=5)

        loss1 = soft_bce.SoftBCEWithLogitsLoss()(
            y_hat1_s, y1) + dice.DiceLoss(mode='binary')(y_hat1_s, y1)
        loss2 = F.cross_entropy(y_hat2, y2) + \
            dice.DiceLoss(mode='multiclass')(y_hat2_s, y2)
        val_loss = (loss1 + loss2) / 2

        # acc1 = FM.accuracy(y_hat1_s, y1.type(torch.int64))
        # acc2 = FM.accuracy(y_hat2_s, y2)
        # val_acc = (acc1 + acc2) / 2

        iou1 = FM.jaccard_index(y_hat1_s, y1, task='binary')
        iou2 = FM.jaccard_index(
            y_hat2_s, y2, task='multiclass', num_classes=12)
        val_iou = (iou1 + iou2) / 2

        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss},
                                           global_step=self.global_step)
        self.logger.experiment.add_scalars('iou', {'val_iou': val_iou},
                                           global_step=self.global_step)
        self.log_dict({'val_loss': val_loss, 'val_iou': val_iou})
        # log images

        # x = resize_right.resize(x, scale_factors=0.5)
        # y_hat1 = resize_right.resize(y_hat1, scale_factors=0.5)
        # y_hat2 = resize_right.resize(y_hat2, scale_factors=0.5)
        # y_hat1 = torch.argmax(y_hat1, 1, keepdim=False)
        # y_hat2 = torch.argmax(y_hat2, 1, keepdim=False)
        # y1_log = apply_colormap(y_hat1.cpu().numpy())
        # y2_log = apply_colormap(y_hat2.cpu().numpy())

        # y1_true = apply_colormap(np.ceil(y1.cpu().numpy()).astype('int64'))
        # y2_true = apply_colormap(np.ceil(y2.cpu().numpy()).astype('int64'))

        # self.logger.experiment.add_images("bscans", x, self.current_epoch)
        # self.logger.experiment.add_images("pred1",
        #                                   y1_log,
        #                                   self.current_epoch,
        #                                   dataformats='NHWC')
        # self.logger.experiment.add_images("gt1",
        #                                   y1_true,
        #                                   self.current_epoch,
        #                                   dataformats='NHWC')
        # self.logger.experiment.add_images("pred2",
        #                                   y2_log,
        #                                   self.current_epoch,
        #                                   dataformats='NHWC')
        # self.logger.experiment.add_images("gt2",
        #                                   y2_true,
        #                                   self.current_epoch,
        #                                   dataformats='NHWC')
        # self.logger.experiment.flush()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)  # 0
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-8,
            verbose=True)
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
        summary(self.to(device), (1, 384, 288))


if __name__ == '__main__':

    model = NetModel(CNNNet)
    model.summary()
    model.to_onnx('test.onnx')
