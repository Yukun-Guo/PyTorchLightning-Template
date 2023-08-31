import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Utils.utils import shuffle_lists, listFiles, k_fold_split, read_img_list_to_npy
from Utils.DataAugmentation import GrayJitter, RandomCrop2D, RandomFlip


class myDataset_mat(Dataset):
    """
        Rewrite this class for your project
    """

    def __init__(self, mat_list, out_size, shuffle=True, img_tag='imgMat', msk_tag='imgMask'):
        if shuffle:
            mat_list = shuffle_lists(mat_list)
            self.mat = shuffle_lists(
                self._read_mat_list_to_npy(mat_list, img_tag, msk_tag))
        else:
            self.mat = self._read_mat_list_to_npy(mat_list, img_tag, msk_tag)
        self.transform = transforms.Compose(
            [GrayJitter(), RandomFlip(axis=1), RandomCrop2D(out_size), Normalize(), ToTensor()])

    @staticmethod
    def _read_mat_list_to_npy(file_list, img_tag, msk_tag):
        npy_data = []
        for f in file_list:
            mat = sio.loadmat(f)
            img, mask = np.array(mat[img_tag], dtype='float32'), np.array(
                mat[msk_tag], dtype='int64')

            npy_data.extend((img[:, :, i], mask[:, :, i])
                            for i in range(img.shape[2]))
            print(f'Reading {f}')
        return npy_data

    def __len__(self):
        return len(self.mat)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img, mask = self.mat[item]
        img = torch.from_numpy(np.expand_dims(img, 0))

        # mask 1: retinal tissue and FV, 0: background
        msk1 = torch.from_numpy(np.clip(mask, 0, 1)).type(torch.float32)
        msk2 = torch.from_numpy(mask)
        sample = {'img': img, 'mask1': msk1, 'mask2': msk2}

        sample = self.transform(sample)
        return sample['img'], (sample['mask1'], sample['mask2'])


class myDataset_img(Dataset):
    """
        Rewrite this class for your project
    """

    def __init__(self, img_list, gt_list, out_size, shuffle=True):
        if shuffle:
            img_list, gt_list = shuffle_lists(img_list, gt_list)

        self.imgs = read_img_list_to_npy(img_list, color_mode='gray')
        self.gts = read_img_list_to_npy(gt_list, color_mode='idx')
        self.transform = transforms.Compose(
            [GrayJitter(), RandomFlip(axis=1), RandomCrop2D(out_size), Normalize(), ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img, mask = self.imgs[item], self.gts[item]

        img = np.expand_dims(img, 0)

        # mask 1: retinal tissue and FV, 0: background
        msk1 = np.clip(mask, 0, 1)
        msk2 = mask
        sample = {'img': img, 'mask1': msk1, 'mask2': msk2}

        sample = self.transform(sample)
        return sample['img'], (sample['mask1'], sample['mask2'])


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, sample):
        img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
        # meanv = torch.mean(img)
        # stdv = torch.std(img)
        # img = F.normalize(img, meanv.numpy(), stdv.numpy(), self.inplace)
        img = img / 255.
        return {'img': img, 'mask1': mask1, 'mask2': mask2}

    def __repr__(self):
        return self.__class__.__name__


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, sample):
        img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # img = torch.from_numpy(img)
        if not torch.is_tensor(mask1):
            mask1 = torch.from_numpy(mask1)
            mask2 = torch.from_numpy(mask2)

        return {'img': img, 'mask1': mask1, 'mask2': mask2}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    img_list = listFiles('data/images', '*.png')
    gt_list = listFiles('data/groundtruth', '*.png')
    file_list = list(zip(img_list, gt_list))
    dataset = myDataset_img(img_list, gt_list, (384, 288))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataloader))

    for img, (mask1, mask2) in dataloader:
        print(img[0].shape)
        print(mask1[0].shape)
        print(mask2[0].shape)
        bscan = make_grid(torch.cat([img, img, img], dim=1)).permute(1, 2, 0)
        plt.figure()
        plt.subplot(1, 3, 1), plt.imshow(bscan.numpy())
        plt.subplot(1, 3, 2), plt.imshow(np.squeeze(mask1[0].numpy()))
        plt.subplot(1, 3, 3), plt.imshow(np.squeeze(mask2[0].numpy()))
        plt.show()
