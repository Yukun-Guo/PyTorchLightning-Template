from typing import Optional
import pytorch_lightning as plt
from torch.utils.data import DataLoader
from PreprocessData import myDataset_img


class DataModel(plt.LightningDataModule):

    def __init__(self, img_list: list, gt_list: list, data_split_idx: list, img_size: tuple = (384, 288), batch_size: int = 32, img_tag='imgMat', msk_tag='imgMask'):
        super().__init__()
        self.img_list = img_list
        self.gt_list = gt_list
        self.data_split_idx = data_split_idx
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_tag = img_tag
        self.msk_tag = msk_tag
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        train_img_list = [self.img_list[i] for i in self.data_split_idx[0]]
        train_gt_list = [self.gt_list[i] for i in self.data_split_idx[0]]
        valid_img_list = [self.img_list[i] for i in self.data_split_idx[1]]
        valid_gt_list = [self.gt_list[i] for i in self.data_split_idx[1]]
        self.train_dataset = myDataset_img(
            train_img_list, train_gt_list, self.img_size)
        self.valid_dataset = myDataset_img(
            valid_img_list, valid_gt_list, self.img_size)
        print(
            f'Train on {len(self.train_dataset)} samples, validation on {len(self.valid_dataset)} samples.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        pass
