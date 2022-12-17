import pytorch_lightning as ptl
from ModuleNet import NetModel
from ModuleData import DataModel
from MyNetwrok import CNNNet
from Utils.utils import listFiles, split_list

ptl.seed_everything(1234)

img_list = listFiles("data\images", '*.png')
gt_list = listFiles("data\groundtruth", '*.png')

file_list = list(zip(img_list, gt_list))

# split file_list into 2 parts: training and validation. training:validation = 3:1

data_split_idx = split_list(file_list, split=(0.7, 0.3))


# data_split_idx = [[3, 7], [0, 1, 2]]

model_path = 'logs\myBackboneNet\myBackboneNet-fold=1-epoch=017-val_loss=1.06065.ckpt'
net_model = NetModel(CNNNet).load_from_checkpoint(model_path)

data_model = DataModel(file_list=file_list,
                       data_split_idx=data_split_idx,
                       img_size=(384, 288),
                       batch_size=8)

trainer = ptl.Trainer(logger=net_model.configure_loggers(),
                      gpus=1,
                      max_epochs=5000,
                      progress_bar_refresh_rate=1,
                      log_every_n_steps=1,
                      auto_lr_find=True)

trainer.validate(model=net_model, datamodule=data_model)
