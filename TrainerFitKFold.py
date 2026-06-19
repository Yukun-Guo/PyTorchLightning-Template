"""
K-fold cross-validation training entry point.

    python TrainerFitKFold.py

Splits the dataset into ``k_fold`` folds and trains one model per fold. Each
fold gets its own logger version and checkpoint filenames (``...-foldN-...``).
Reuses the same config and best-practice Trainer as :mod:`TrainerFit`.
"""

import lightning as L
from torch.utils.data import DataLoader

from DataPreprocessing import SegmentationDataset
from NetModule import NetModule
from Utils.training import build_trainer, load_config
from Utils.utils import k_fold_split, listFiles


class KFoldDataModule(L.LightningDataModule):
    """DataModule for a single fold with explicit train/val file lists."""

    def __init__(self, train_files, val_files, img_size, channels, batch_size,
                 num_workers, shuffle, augmentation):
        super().__init__()
        self.train_imgs, self.train_gts = train_files
        self.val_imgs, self.val_gts = val_files
        self.img_size = img_size
        self.channels = channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.augmentation = augmentation

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            self.train_imgs, self.train_gts, self.img_size, channels=self.channels,
            train=True, augmentation=self.augmentation,
        )
        self.val_dataset = SegmentationDataset(
            self.val_imgs, self.val_gts, self.img_size, channels=self.channels,
            train=False, augmentation=False,
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0, drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.shuffle)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)


def main(config_path: str = "config.toml"):
    config = load_config(config_path)
    L.seed_everything(config["Project"]["seed"], workers=True)

    data_cfg = config["DataModule"]
    img_list = listFiles(data_cfg["image_path"], "*.png")
    gt_list = listFiles(data_cfg["mask_path"], "*.png")
    if not img_list or len(img_list) != len(gt_list):
        raise ValueError("Images and masks must exist and have equal counts.")

    k_folds = data_cfg.get("k_fold", 5)
    channels = data_cfg["image_shape"][0]
    img_size = tuple(data_cfg["image_shape"][1:])

    _, fold_indices = k_fold_split(list(range(len(img_list))), fold=k_folds)

    print(f"Starting {k_folds}-fold cross validation on {len(img_list)} samples.")
    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        print(f"\n===== Fold {fold}/{k_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) =====")

        datamodule = KFoldDataModule(
            train_files=([img_list[i] for i in train_idx], [gt_list[i] for i in train_idx]),
            val_files=([img_list[i] for i in val_idx], [gt_list[i] for i in val_idx]),
            img_size=img_size,
            channels=channels,
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=data_cfg.get("shuffle", True),
            augmentation=data_cfg.get("augmentation", True),
        )

        model = NetModule(config=config)
        trainer = build_trainer(config, fold=fold)
        trainer.fit(model, datamodule=datamodule)

    print("\nK-fold cross validation completed.")


if __name__ == "__main__":
    main()
