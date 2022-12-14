import pytorch_lightning as ptl
from ModuleNet import NetModel
from ModuleData import DataModel
from MyNetwrok import CNNNet
from Utils.utils import listFiles

ptl.seed_everything(1234)

file_list = listFiles("F:\Data4LayerSegmentation\_Dataset_v2_", '*.mat')

data_split_idx = [[
    0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 24, 25, 26,
    27, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
], [3, 7, 11, 15, 19, 22, 27, 31, 35, 39, 41, 45, 49]]
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
