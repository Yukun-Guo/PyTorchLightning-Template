import pytorch_lightning as plt
from torch.utils.data import DataLoader
from PreprocessData import myDataset_mat


class DataModule(plt.LightningDataModule):

    def __init__(self, file_list: list, data_split_idx: list, img_size: tuple = (384, 288), batch_size: int = 32):
        super().__init__()
        self.file_list = file_list
        self.data_split_idx = data_split_idx
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        pass

    def prepare_data(self):
        train_list = [self.file_list[i] for i in self.data_split_idx[0]]
        valid_list = [self.file_list[i] for i in self.data_split_idx[1]]
        self.train_dataset = myDataset_mat(train_list, self.input_size)
        self.valid_dataset = myDataset_mat(valid_list, self.input_size)
        print(
            f'Train on {len(self.train_dataset)} samples, validation on {len(self.valid_dataset)} samples.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
