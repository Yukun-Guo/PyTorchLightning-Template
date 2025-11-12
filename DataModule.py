"""
PyTorch Lightning DataModule for image segmentation tasks.

This module provides a comprehensive data handling solution for image segmentation,
including data loading, preprocessing, splitting, and DataLoader creation.
"""

import lightning as L
from typing import Optional
from torch.utils.data import DataLoader
from DataPreprocessing import myDataset_img
from Utils.utils import listFiles, split_list


class DataModel(L.LightningDataModule):
    """
    Lightning DataModule for image segmentation data handling.
    
    This class manages all data-related operations including:
    - Loading image and mask file lists
    - Splitting data into train/validation/test sets
    - Creating dataset instances with appropriate transforms
    - Providing DataLoaders for training, validation, and testing
    
    Args:
        config (dict): Configuration dictionary containing DataModule settings.
            Required keys:
            - image_path (str): Path to directory containing input images
            - mask_path (str): Path to directory containing ground truth masks
            - batch_size (int): Batch size for DataLoaders
            - image_shape (list): Target image dimensions [height, width, channels]
            - shuffle (bool): Whether to shuffle training data
            - split_ratio (list): Data split ratios [train, test, validation]
    
    Attributes:
        image_path (str): Path to input images directory
        mask_path (str): Path to ground truth masks directory
        batch_size (int): Batch size for data loaders
        img_shape (list): Image dimensions [height, width, channels]
        shuffle (bool): Whether to shuffle training data
        split_ratio (list): Data split ratios
        img_size (tuple): Image size as (height, width)
        train_dataset: Training dataset instance
        valid_dataset: Validation dataset instance
        test_dataset: Test dataset instance
    
    Example:
        >>> config = {
        ...     'DataModule': {
        ...         'image_path': './data/images',
        ...         'mask_path': './data/masks',
        ...         'batch_size': 16,
        ...         'image_shape': [256, 256, 1],
        ...         'shuffle': True,
        ...         'split_ratio': [0.7, 0.2, 0.1]
        ...     }
        ... }
        >>> data_module = DataModel(config)
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
    """

    def __init__(self, config: dict):
        """
        Initialize the DataModel with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with DataModule settings
        """
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
        self.test_dataset = None

    def prepare_data(self):
        """
        Prepare data for training (called only on 1 GPU/TPU in distributed training).
        
        This method is called only once and is used for downloading or preparing
        data that should not be done in parallel. Currently not implemented as
        data preparation is handled in setup().
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        
        This method is called on every GPU in distributed training and is responsible
        for loading data files, splitting them according to the configuration, and
        creating dataset instances.
        
        Args:
            stage (str, optional): Current stage of training. Can be 'fit', 'validate',
                'test', or None. Defaults to None.
        
        Side Effects:
            - Sets self.train_dataset, self.valid_dataset, and self.test_dataset
            - Prints dataset sizes to console
        
        Raises:
            FileNotFoundError: If image or mask directories don't exist
            ValueError: If number of images and masks don't match
        """
        self.img_list = listFiles(self.image_path, "*.png")
        self.gt_list = listFiles(self.mask_path, "*.png")
        
        if len(self.img_list) != len(self.gt_list):
            raise ValueError(f"Mismatch: {len(self.img_list)} images vs {len(self.gt_list)} masks")
        
        self.train_list, self.test_list, self.valid_list = split_list(
            list(range(len(self.img_list))), split=self.split_ratio
        )
        
        train_img_list = [self.img_list[i] for i in self.train_list]
        train_gt_list = [self.gt_list[i] for i in self.train_list]
        valid_img_list = [self.img_list[i] for i in self.valid_list]
        valid_gt_list = [self.gt_list[i] for i in self.valid_list]
        test_img_list = [self.img_list[i] for i in self.test_list]
        test_gt_list = [self.gt_list[i] for i in self.test_list]
        
        self.train_dataset = myDataset_img(train_img_list, train_gt_list, self.img_size)
        self.valid_dataset = myDataset_img(valid_img_list, valid_gt_list, self.img_size)
        self.test_dataset = myDataset_img(test_img_list, test_gt_list, self.img_size)
        
        print(f'Train on {len(self.train_dataset)} samples, '
              f'validation on {len(self.valid_dataset)} samples, '
              f'test on {len(self.test_dataset)} samples.')

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the training DataLoader.
        
        Returns:
            DataLoader: Training data loader with shuffling enabled
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation DataLoader.
        
        Returns:
            DataLoader: Validation data loader without shuffling
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test DataLoader.
        
        Returns:
            DataLoader: Test data loader without shuffling
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def teardown(self, stage: Optional[str] = None):
        """
        Clean up after training/testing is complete.
        
        Args:
            stage (str, optional): Current stage being torn down
        """
        pass
