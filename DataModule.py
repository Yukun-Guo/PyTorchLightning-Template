"""
PyTorch Lightning DataModule for image segmentation.

Responsibilities:
  - discover image / mask files
  - split them into train / val / test sets (or load a custom split)
  - build :class:`SegmentationDataset` instances with the right transforms
  - expose train / val / test DataLoaders

To adapt to your own data you usually only edit ``config.toml``. For a custom
on-disk layout (different folders, file types, paired naming), adjust
:meth:`setup` and the file-listing logic below.
"""

from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from DataPreprocessing import SegmentationDataset
from Utils.utils import listFiles, split_list


class DataModel(L.LightningDataModule):
    """LightningDataModule driven entirely by the ``[DataModule]`` config section."""

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["DataModule"]
        self.image_path = cfg["image_path"]
        self.mask_path = cfg["mask_path"]
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg.get("num_workers", 4)
        self.shuffle = cfg.get("shuffle", True)
        self.augmentation = cfg.get("augmentation", True)
        self.split_ratio = cfg["split_ratio"]  # [train, val, test]

        # image_shape is (C, H, W)
        self.channels = cfg["image_shape"][0]
        self.img_size = tuple(cfg["image_shape"][1:])

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        img_list = listFiles(self.image_path, "*.png")
        gt_list = listFiles(self.mask_path, "*.png")

        if not img_list or not gt_list:
            raise FileNotFoundError(
                f"No images/masks found in '{self.image_path}' / '{self.mask_path}'."
            )
        if len(img_list) != len(gt_list):
            raise ValueError(f"Mismatch: {len(img_list)} images vs {len(gt_list)} masks")

        indices = list(range(len(img_list)))
        train_idx, val_idx, test_idx = split_list(indices, split=self.split_ratio)

        def subset(idx):
            return [img_list[i] for i in idx], [gt_list[i] for i in idx]

        train_imgs, train_gts = subset(train_idx)
        val_imgs, val_gts = subset(val_idx)
        test_imgs, test_gts = subset(test_idx)

        self.train_dataset = SegmentationDataset(
            train_imgs, train_gts, self.img_size, channels=self.channels,
            train=True, augmentation=self.augmentation,
        )
        self.valid_dataset = SegmentationDataset(
            val_imgs, val_gts, self.img_size, channels=self.channels,
            train=False, augmentation=False,
        )
        self.test_dataset = SegmentationDataset(
            test_imgs, test_gts, self.img_size, channels=self.channels,
            train=False, augmentation=False,
        )

        print(
            f"Train on {len(self.train_dataset)} samples, "
            f"validate on {len(self.valid_dataset)} samples, "
            f"test on {len(self.test_dataset)} samples."
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle,  # drop last partial batch only when training
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
