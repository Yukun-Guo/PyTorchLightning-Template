import os
import lightning as L
import torch
from NetModule import NetModule
from DataModule import DataModel
import toml
# set device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.device_count())
L.seed_everything(1234)

toml_file = "./config.toml"
config = toml.load(toml_file)

data_model = DataModel(config=config)

net_model = NetModule(config=config)

trainer = L.Trainer(
    logger=net_model.configure_loggers(),
    accelerator="gpu",
    devices=[0],
    max_epochs=5000,
   # strategy="auto",  # strategy='ddp_sharded', # model parallelism,
    log_every_n_steps=None,
)

trainer.fit(net_model, datamodule=data_model)
