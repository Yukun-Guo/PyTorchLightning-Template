import pytorch_lightning as ptl
from ModuleNet import NetModel
from MyNetwrok import CNNNet
from Utils.utils import listFiles
ptl.seed_everything(1234)
folder = r"F:\Data4LayerSegmentation\_Dataset_v2_"
file_list = listFiles(folder, '*.mat')

data_split_idx = [[3, 7, 11, 15, 19, 22, 27, 31, 35, 39, 41, 45, 49],
                  [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18,
                   19, 21, 22, 24, 25, 26, 27, 29, 30, 31, 33, 34,
                   35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53,
                   54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]]
model = NetModel(backbone_net=CNNNet, file_list=file_list, batch_size=8, input_size=(
    384, 288), data_split_idx=data_split_idx)
print(model)
trainer = ptl.Trainer(logger=model.configure_loggers(), gpus=1, max_epochs=5000,
                      progress_bar_refresh_rate=1, log_every_n_steps=1, auto_lr_find=True)
trainer.fit(model)
