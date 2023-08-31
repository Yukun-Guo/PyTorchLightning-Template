import lightning as L
from NetModule import NetModule
from DataModule import DataModel
from Network import CNNNet
import numpy as np
from Utils.utils import listFiles, k_fold_split

L.seed_everything(1234)

fold = 4

img_list = listFiles("data\images", "*.png")
gt_list = listFiles("data\groundtruth", "*.png")

# kFold_list_list, kFold_list_idx = k_fold_split(img_list, fold=fold)


# print(kFold_list_list)

# for data_split_idx in kFold_list_idx:
#     net_model = NetModule(backbone_net=CNNNet, input_size=(384, 288))

#     data_model = DataModel(
#         img_list, gt_list, data_split_idx=(data_split_idx[0], data_split_idx[1]), img_size=(384, 288), batch_size=8
#     )
#     trainer = L.Trainer(
#         logger=net_model.configure_loggers(),
#         accelerator="gpu",
#         devices=[0],
#         max_epochs=5000,
#     # strategy="auto",  # strategy='ddp_sharded', # model parallelism,
#         log_every_n_steps=1,
#     )
#     trainer.fit(net_model, datamodule=data_model)
