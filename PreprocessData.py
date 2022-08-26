import os
import torch
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Utils.utils import shuffle_lists, listFiles, k_fold_split


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
            [GrayJitter(), RandomHorizontalFlip(), RandomCrop(out_size), Normalize(), ToTensor()])

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
        msk1 = torch.from_numpy(np.clip(mask, 0, 1))
        msk2 = torch.from_numpy(mask)
        sample = {'img': img, 'mask1': msk1, 'mask2': msk2}

        sample = self.transform(sample)
        return sample['img'], (sample['mask1'], sample['mask2'])


class GrayJitter(object):
    def __init__(self, bright_range=(0, 40), contrast_range=(0.5, 1.5), max_value=255):
        self.bright_range = bright_range
        self.contrast_range = contrast_range
        self.max_value = max_value

    def __call__(self, sample):
        img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
        bright_scale = random.uniform(
            self.bright_range[0], self.bright_range[1])
        contrast_scale = random.uniform(
            self.contrast_range[0], self.contrast_range[1])
        meanv = torch.mean(img)
        img = (img - meanv) * contrast_scale + meanv
        img = img + bright_scale
        img = torch.clip(img, 0, self.max_value)
        return {'img': img, 'mask1': mask1, 'mask2': mask2}


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=5.):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        img, ilm, thk, msk = sample['img'], sample['ilm'], sample['thk'], sample['msk']
        img = torch.clip(img + torch.randn(img.size())
                         * self.std + self.mean, 0, 255)
        ilm = torch.clip(ilm + torch.randn(ilm.size())
                         * 1.1 + self.mean, 0, 255)
        thk = torch.clip(thk + torch.randn(thk.size())
                         * 1.1 + self.mean, 0, 255)

        return {'img': img, 'ilm': ilm, 'thk': thk, 'msk': msk}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomRotate90n(object):
    """Rotate the given PIL Image randomly with a given probability.

    Args:

    """

    def __call__(self, sample):
        image, mask = sample['img'], sample['mask']
        degree = random.randint(0, 3)
        image = F.rotate(image, 90 * degree)
        mask = torch.squeeze(F.rotate(torch.unsqueeze(mask, 0), 90 * degree))
        return {'img': image, 'mask': mask}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
        if random.random() < self.p:
            image = F.hflip(image)
            mask1 = F.hflip(mask1)
            mask2 = F.hflip(mask2)
        return {'img': image, 'mask1': mask1, 'mask2': mask2}


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return {'img': image, 'mask': mask}


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        output_size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, output_size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.size = output_size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size, mask=None):
        w, h = img.shape[-2:][::-1]
        th, tw = output_size
        range_w = 1 if w == tw else w - tw
        if h == th:
            range_h = 1
        elif mask is not None:
            rowline = torch.sum(mask, (1, ))
            if torch.sum(rowline) != 0:
                idx = torch.nonzero(rowline)
                off = torch.tensor(h - th)
                range_h = torch.min(off, idx[0]).numpy()[0]
            else:
                range_h = 1
        else:
            range_h = h - th
        i = random.randint(0, range_h)
        j = random.randint(0, range_w)
        return i, j, th, tw

    def __call__(self, sample):
        img, mask1, mask2 = sample['img'], sample['mask1'], sample['mask2']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask1 = F.pad(mask1, self.padding, self.fill, self.padding_mode)
            mask2 = F.pad(mask2, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        size = img.shape[-2:][::-1]
        if self.pad_if_needed and size[0] < self.size[1]:
            img = F.pad(img, [self.size[1] - size[0], 0],
                        self.fill, self.padding_mode)
            mask1 = F.pad(mask1, [self.size[1] - size[0],
                          0], self.fill, self.padding_mode)
            mask2 = F.pad(mask2, [self.size[1] - size[0],
                          0], self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and size[1] < self.size[0]:
            img = F.pad(img, [0, self.size[0] - size[1]],
                        self.fill, self.padding_mode)
            mask1 = F.pad(mask1, [0, self.size[0] - size[1]],
                          self.fill, self.padding_mode)
            mask2 = F.pad(mask2, [0, self.size[0] - size[1]],
                          self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size, mask1)

        img = F.crop(img, i, j, h, w)
        mask1 = F.crop(mask1, i, j, h, w)
        mask2 = F.crop(mask2, i, j, h, w)

        return {'img': img, 'mask1': mask1, 'mask2': mask2}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


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

    file_list = listFiles('F:\Data4LayerSegmentation\_Dataset_v2_', '*.mat')
    lists, idxs = k_fold_split(file_list[:10], 5)

    print(lists)
    print(idxs)

    dataset = myDataset_mat(
        lists[0][1], (384, 288), img_tag='imgMat', msk_tag='imgMask')
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
