"""
Data preprocessing and dataset classes for image segmentation.

This module provides dataset classes and transformation utilities for loading
and preprocessing image segmentation data, including data augmentation and
format conversion.
"""

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union

from Utils.utils import shuffle_lists, listFiles, read_img_list_to_npy
from Utils.DataAugmentation import GrayJitter, RandomCrop2D, RandomFlip


class myDataset_img(Dataset):
    """
    Custom Dataset class for image segmentation tasks.
    
    This dataset handles loading and preprocessing of image-mask pairs for
    semantic segmentation. It supports data augmentation and automatic
    format conversion.
    
    Args:
        img_list (list): List of file paths to input images
        gt_list (list): List of file paths to ground truth masks
        out_size (tuple): Target output size (height, width) for images
        shuffle (bool, optional): Whether to shuffle image-mask pairs. Defaults to True.
    
    Attributes:
        imgs (list): List of loaded image arrays
        gts (list): List of loaded ground truth arrays
        transform (transforms.Compose): Composition of data transforms
    
    Note:
        - Images are loaded as grayscale (single channel)
        - Masks are loaded as indexed images with integer class labels
        - Data augmentation includes grayscale jittering, random flip, and random crop
        - All samples are normalized to [0, 1] range and converted to tensors
    
    Example:
        >>> img_files = ['img1.png', 'img2.png']
        >>> mask_files = ['mask1.png', 'mask2.png']
        >>> dataset = myDataset_img(img_files, mask_files, (256, 256))
        >>> image, mask = dataset[0]
        >>> print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    """

    def __init__(self, img_list: List[str], gt_list: List[str], out_size: Tuple[int, int], shuffle: bool = True):
        """
        Initialize the dataset with image and mask file lists.
        
        Args:
            img_list (list): List of paths to input images
            gt_list (list): List of paths to ground truth masks
            out_size (tuple): Target output size (height, width)
            shuffle (bool): Whether to shuffle the data pairs
        """
        if shuffle:
            img_list, gt_list = shuffle_lists(img_list, gt_list)

        self.imgs = read_img_list_to_npy(img_list, color_mode='gray')
        self.gts = read_img_list_to_npy(gt_list, color_mode='idx')
        self.transform = transforms.Compose([
            GrayJitter(),
            RandomFlip(axis=1),
            RandomCrop2D(out_size),
            Normalize(),
            ToTensor()
        ])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of image-mask pairs in the dataset
        """
        return len(self.imgs)

    def __getitem__(self, item: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            item (int or torch.Tensor): Index of the sample to retrieve
        
        Returns:
            tuple: (image_tensor, mask_tensor) where:
                - image_tensor: Preprocessed image tensor of shape (C, H, W)
                - mask_tensor: Ground truth mask tensor of shape (H, W)
        """
        if torch.is_tensor(item):
            item = item.tolist()

        img, mask = self.imgs[item], self.gts[item]
        img = np.expand_dims(img, 0)  # Add channel dimension

        sample = {'img': img, 'mask': mask}
        sample = self.transform(sample)
        
        return sample['img'], sample['mask']

class Normalize(object):
    """
    Normalize image pixel values from [0, 255] to [0, 1] range.
    
    This transform converts uint8 pixel values to float32 values in the [0, 1] range
    by dividing by 255. The mask values are left unchanged as they represent
    class indices.
    
    Args:
        inplace (bool, optional): Whether to perform normalization in-place.
            Defaults to False.
    
    Example:
        >>> normalize = Normalize()
        >>> sample = {'img': np.array([0, 128, 255]), 'mask': np.array([0, 1, 2])}
        >>> normalized = normalize(sample)
        >>> print(normalized['img'])  # [0.0, 0.502, 1.0]
    """

    def __init__(self, inplace: bool = False):
        """
        Initialize the Normalize transform.
        
        Args:
            inplace (bool): Whether to perform normalization in-place
        """
        self.inplace = inplace

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply normalization to the sample.
        
        Args:
            sample (dict): Dictionary containing 'img' and 'mask' arrays
        
        Returns:
            dict: Dictionary with normalized image and original mask
        """
        img, mask = sample['img'], sample['mask']
        img = img / 255.0  # Normalize to [0, 1]
        return {'img': img, 'mask': mask}

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        return self.__class__.__name__


class ToTensor(object):
    """
    Convert numpy arrays to PyTorch tensors.
    
    This transform converts numpy arrays to PyTorch tensors with appropriate
    data types:
    - Images: Converted to FloatTensor (for gradient computation)
    - Masks: Converted to LongTensor (for class indices)
    
    The transform maintains the array dimensions and does not change the
    data layout (assumes images are already in C x H x W format).
    
    Example:
        >>> to_tensor = ToTensor()
        >>> sample = {
        ...     'img': np.random.rand(1, 256, 256).astype(np.float32),
        ...     'mask': np.random.randint(0, 4, (256, 256)).astype(np.int64)
        ... }
        >>> tensor_sample = to_tensor(sample)
        >>> print(type(tensor_sample['img']))  # <class 'torch.Tensor'>
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert sample arrays to tensors.
        
        Args:
            sample (dict): Dictionary containing 'img' and 'mask' numpy arrays
        
        Returns:
            dict: Dictionary with tensor versions of image and mask
        """
        img, mask = sample['img'], sample['mask']
        
        # Convert to tensors with appropriate dtypes
        if not torch.is_tensor(img):
            img = torch.from_numpy(img.astype(np.float32))
        
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask.astype(np.int64))

        return {'img': img, 'mask': mask}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    img_list = listFiles('data/images', '*.png')
    gt_list = listFiles('data/groundtruth', '*.png')
    file_list = list(zip(img_list, gt_list))
    dataset = myDataset_img(img_list, gt_list, (384, 288))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataloader))

    for img, mask in dataloader:
        print(img[0].shape)
        print(mask[0].shape)
        bscan = make_grid(torch.cat([img, img, img], dim=1)).permute(1, 2, 0)
        plt.figure()
        plt.subplot(1, 3, 1), plt.imshow(bscan.numpy())
        plt.subplot(1, 3, 2), plt.imshow(np.squeeze(mask[0].numpy()))
        plt.subplot(1, 3, 3), plt.imshow(np.squeeze(mask[0].numpy()))
        plt.show()
