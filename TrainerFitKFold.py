import pytorch_lightning as ptl
from CNN_2D import SegModel
from Dataset_processing import read_file_list, k_fold_split

fold = 4
ptl.seed_everything(1234)

file_list = r"F:\Data4LayerSegmentation\_Dataset\_ArrangedData\sampledDataset\_mat_list.txt"
files = read_file_list(file_list)

kFold_list_list, kFold_list_idx = k_fold_split(files, fold=fold)

print(kFold_list_list)

for i, data_split_idx in enumerate(kFold_list_idx):
    model = SegModel(file_list, batch_size=32, input_size=(
        384, 288), data_split_idx=data_split_idx, k_fold=i)
    trainer = ptl.Trainer(logger=model.configure_loggers(), gpus=1, max_epochs=5000,
                          progress_bar_refresh_rate=1, log_every_n_steps=1, auto_lr_find=True)
    trainer.fit(model)
