import pytorch_lightning as ptl
from ModuleNet import NetModel
from ModuleData import DataModel
from MyNetwrok import CNNNet
import numpy as np
from Utils.utils import listFiles, split_list
ptl.seed_everything(1234)


img_list = listFiles("data\images", '*.png')
gt_list = listFiles("data\groundtruth", '*.png')

data_split_idx = split_list(list(range(len(img_list))), split=(0.7, 0.3))

net_model = NetModel(backbone_net=CNNNet, input_size=(384, 288))

data_model = DataModel(img_list, gt_list, data_split_idx=(data_split_idx[0], data_split_idx[1]), img_size=(
    384, 288), batch_size=8)

trainer = ptl.Trainer(logger=net_model.configure_loggers(),
                      accelerator='gpu',
                      devices=2,
                      max_epochs=5000,
                      strategy='dp',  # strategy='ddp_sharded', # model parallelism,
                      log_every_n_steps=1,
                      auto_lr_find=True)

trainer.fit(net_model, datamodule=data_model)
