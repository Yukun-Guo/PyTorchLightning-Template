"""
Dataset and preprocessing transforms for image segmentation.

A sample flows through the pipeline as a ``{"img": ..., "mask": ...}`` dict:
  1. images / masks are read into numpy arrays (img is channel-first ``(C, H, W)``)
  2. optional augmentations are applied (training split only)
  3. ``Normalize`` scales pixels to ``[0, 1]``
  4. ``ToTensor`` converts to tensors
  5. ``Resize`` guarantees every sample has the same ``(H, W)`` for batching

To change the augmentations, edit :func:`build_transforms`. To support a
different on-disk format (e.g. ``.npy`` volumes, RGB images), adjust how
:class:`SegmentationDataset` reads files.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from Utils.DataAugmentation import GrayJitter
from Utils.utils import read_img_list_to_npy, shuffle_lists


class RandomFlip:
    """Randomly flip image and mask together along a spatial axis.

    Works whether the image is channel-first ``(C, H, W)`` or 2-D ``(H, W)``:
    ``axis`` indexes from the end (``-1`` = horizontal / W, ``-2`` = vertical / H).
    """

    def __init__(self, p: float = 0.5, axis: int = -1):
        self.p = p
        self.axis = axis

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.random() < self.p:
            img = np.flip(sample["img"], axis=self.axis)
            mask = np.flip(sample["mask"], axis=self.axis)
            return {"img": img, "mask": mask}
        return sample


class Normalize:
    """Scale image pixels from ``[0, 255]`` to ``[0, 1]``; masks are untouched."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {"img": sample["img"] / 255.0, "mask": sample["mask"]}


class ToTensor:
    """Convert numpy arrays to tensors (float image, long mask)."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        img, mask = sample["img"], sample["mask"]
        if not torch.is_tensor(img):
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return {"img": img, "mask": mask}


class Resize:
    """Resize image (bilinear) and mask (nearest) to a fixed ``(H, W)`` size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = tuple(size)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img, mask = sample["img"], sample["mask"]
        if tuple(img.shape[-2:]) != self.size:
            img = F.interpolate(img.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
            mask = (
                F.interpolate(mask[None, None].float(), size=self.size, mode="nearest")
                .squeeze(0)
                .squeeze(0)
                .long()
            )
        return {"img": img, "mask": mask}


def build_transforms(out_size: Tuple[int, int], train: bool, augmentation: bool) -> transforms.Compose:
    """Compose the transform pipeline for the train or eval split.

    >>> EDIT HERE to add / remove augmentations. <<<
    Augmentations are applied to the training split only.
    """
    pipeline = []
    if train and augmentation:
        pipeline += [
            GrayJitter(),
            RandomFlip(axis=-1),  # horizontal flip (W axis)
        ]
    pipeline += [Normalize(), ToTensor(), Resize(out_size)]
    return transforms.Compose(pipeline)


class SegmentationDataset(Dataset):
    """Image/mask segmentation dataset.

    Args:
        img_list: paths to input images.
        gt_list: paths to ground-truth masks (indexed PNGs, one int per class).
        out_size: target ``(H, W)`` size.
        channels: number of input channels (1 = grayscale, 3 = RGB).
        train: if True, apply augmentations (when ``augmentation`` is set).
        augmentation: master switch for augmentations.
        shuffle: shuffle the image/mask pairs once at construction time.
    """

    def __init__(
        self,
        img_list: List[str],
        gt_list: List[str],
        out_size: Tuple[int, int],
        channels: int = 1,
        train: bool = True,
        augmentation: bool = True,
        shuffle: bool = True,
    ):
        if shuffle:
            img_list, gt_list = shuffle_lists(img_list, gt_list)

        color_mode = "gray" if channels == 1 else "rgb"
        self.imgs = read_img_list_to_npy(img_list, color_mode=color_mode)
        self.gts = read_img_list_to_npy(gt_list, color_mode="idx")
        self.channels = channels
        self.transform = build_transforms(out_size, train=train, augmentation=augmentation)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(item):
            item = item.tolist()

        img, mask = self.imgs[item], self.gts[item]
        # Move to channel-first: (H, W) -> (1, H, W) or (H, W, 3) -> (3, H, W)
        img = img[None, ...] if img.ndim == 2 else np.transpose(img, (2, 0, 1))

        sample = self.transform({"img": img, "mask": mask})
        return sample["img"], sample["mask"]


# Backwards-compatible alias (the old class name).
myDataset_img = SegmentationDataset
