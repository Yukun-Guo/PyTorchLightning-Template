import lightning as L
from typing import Optional
from torch.utils.data import DataLoader
from DataPreprocessing import myDataset_img
from Utils.utils import listFiles, split_list

class DataModel(L.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.image_path = config['DataModule']["image_path"]
        self.mask_path = config['DataModule']["mask_path"]
        self.batch_size = config['DataModule']["batch_size"]
        self.img_shape = config['DataModule']["image_shape"]
        self.shuffle = config['DataModule']["shuffle"]
        self.split_ratio = config['DataModule']["split_ratio"]
        self.img_size = self.img_shape[:2]  # (H, W)
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.img_list = listFiles(self.image_path, "*.png")
        self.gt_list = listFiles(self.mask_path, "*.png")
        self.train_list, self.test_list,self.valid_list = split_list(list(range(len(self.img_list))), split=self.split_ratio)
        
        train_img_list = [self.img_list[i] for i in self.train_list]
        train_gt_list = [self.gt_list[i] for i in self.train_list]
        valid_img_list = [self.img_list[i] for i in self.valid_list]
        valid_gt_list = [self.gt_list[i] for i in self.valid_list]
        test_img_list = [self.img_list[i] for i in self.test_list]
        test_gt_list = [self.gt_list[i] for i in self.test_list]
        self.train_dataset = myDataset_img( train_img_list, train_gt_list, self.img_size)
        self.valid_dataset = myDataset_img(valid_img_list, valid_gt_list, self.img_size)
        self.test_dataset = myDataset_img(test_img_list, test_gt_list, self.img_size)
        print(f'Train on {len(self.train_dataset)} samples, validation on {len(self.valid_dataset)} samples., test on {len(self.test_dataset)} samples.')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        pass
