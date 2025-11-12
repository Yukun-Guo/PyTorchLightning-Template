"""
Simplified K-Fold Cross Validation Training Script

This script performs k-fold cross validation training using the project's
Lightning model and data pipeline.
"""

import toml
import lightning as L
from torch.utils.data import DataLoader
from DataPreprocessing import myDataset_img
from NetModule import NetModule
from Utils.utils import k_fold_split, listFiles


class KFoldDataModule(L.LightningDataModule):
    """Simple DataModule for K-fold training with explicit train/val splits."""
    
    def __init__(self, train_img_list, train_gt_list, val_img_list, val_gt_list, 
                 img_size, batch_size, shuffle=True):
        super().__init__()
        self.train_img_list = train_img_list
        self.train_gt_list = train_gt_list
        self.val_img_list = val_img_list
        self.val_gt_list = val_gt_list
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = myDataset_img(
            self.train_img_list, self.train_gt_list, self.img_size
        )
        self.val_dataset = myDataset_img(
            self.val_img_list, self.val_gt_list, self.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )


def run_kfold_training():
    """Main function to run k-fold cross validation training."""
    # Set random seed
    L.seed_everything(1234)
    
    # Load configuration
    with open('config.toml', 'r') as f:
        config = toml.load(f)
    
    # Get configuration parameters
    data_config = config['DataModule']
    net_config = config['NetModule']
    
    # Load image and mask files
    img_list = listFiles(data_config['image_path'], "*.png")
    gt_list = listFiles(data_config['mask_path'], "*.png")
    
    if not img_list or not gt_list:
        raise ValueError("No images or masks found in specified directories")
    
    if len(img_list) != len(gt_list):
        raise ValueError("Number of images and masks must be equal")
    
    # K-fold parameters
    k_folds = data_config.get('k_fold', 5)
    img_size = tuple(data_config['image_shape'][:2])  # (H, W)
    batch_size = data_config['batch_size']
    
    # Generate k-fold splits
    _, kfold_indices = k_fold_split(img_list, fold=k_folds)
    
    print(f"Starting {k_folds}-fold cross validation")
    print(f"Total samples: {len(img_list)}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print("-" * 50)
    
    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold_indices):
        print(f"Training Fold {fold_idx + 1}/{k_folds}")
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        # Create file lists for this fold
        train_img_list = [img_list[i] for i in train_indices]
        train_gt_list = [gt_list[i] for i in train_indices]
        val_img_list = [img_list[i] for i in val_indices]
        val_gt_list = [gt_list[i] for i in val_indices]
        
        # Create data module for this fold
        datamodule = KFoldDataModule(
            train_img_list=train_img_list,
            train_gt_list=train_gt_list,
            val_img_list=val_img_list,
            val_gt_list=val_gt_list,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=data_config.get('shuffle', True)
        )
        
        # Update config with current fold info
        fold_config = config.copy()
        fold_config['DataModule']['k_fold'] = fold_idx + 1
        
        # Create model
        model = NetModule(config=fold_config)
        
        # Create trainer
        import torch
        use_gpu = torch.cuda.is_available()
        
        trainer = L.Trainer(
            logger=model.configure_loggers(),
            callbacks=model.configure_callbacks(),
            devices=1 if use_gpu else 0,
            accelerator='gpu' if use_gpu else 'cpu',
            max_epochs=net_config.get('epochs', 500),
            log_every_n_steps=1,
            enable_progress_bar=True
        )
        
        # Train the model
        trainer.fit(model, datamodule=datamodule)
        
        print(f"Fold {fold_idx + 1} completed")
        print("-" * 50)
    
    print("K-fold cross validation completed!")


if __name__ == "__main__":
    run_kfold_training()

