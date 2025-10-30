import numpy as np
import scipy.io as sio
import PIL.Image as Image
import glob
import os
from tqdm import tqdm

def read_file_list(list_txt_file):
    fp = open(list_txt_file, 'r')
    files = fp.readlines()
    files = [item.rstrip() for item in files]
    return files


def listFiles(folder, file_filter="**/*", recursive=True):
    return list(
        glob.iglob(os.path.join(folder, file_filter), recursive=recursive))


def split_list(file_list, split=(0.8, 0.2, 0), shuffle=True):
    if shuffle:
        np.random.shuffle(file_list)
    image_num = len(file_list)
    split_idx = [int(np.ceil(element * image_num)) for element in split]
    s = sum(split_idx[:2])
    train_list = file_list[:split_idx[0]]
    test_list = file_list[split_idx[0]:s]
    valid_list = file_list[s:]
    return train_list, test_list, valid_list


def k_fold_split(file_list, fold=5):
    kFold_list_file = []
    kFold_list_idx = []
    data_set_length = len(file_list)
    index = np.tile(np.arange(0, fold),
                    data_set_length // fold + 1)[:data_set_length]
    for i in range(fold):
        flg_valid = np.where(index == i)[0]
        flg_train = np.where(index != i)[0]
        train_lst = [file_list[j] for j in flg_valid]
        Valid_lst = [file_list[j] for j in flg_train]
        kFold_list_idx.append([flg_train, flg_valid])
        kFold_list_file.append([train_lst, Valid_lst])
    return kFold_list_file, kFold_list_idx


def shuffle_lists(*file_lists):
    """
    Example:
        list1=['a', 'b', 'c']
        list2=[1, 2, 3]
        list1_s,list2_s=shuffle_data_files(list1,list2)
        list1_s = ['a', 'c', 'b']
        list2_s = [1, 3, 2]
    :param file_lists: any numbers of list
    :return: shuffled lists
    """
    if len(file_lists) == 1:
        list_files = list(*file_lists)
        np.random.shuffle(list_files)
        return list_files
    else:
        list_files = list(zip(*file_lists))
        np.random.shuffle(list_files)
        return zip(*list_files)


def read_mat_list_to_npy2d(file_list, shuffle=True):
    npy_data = []
    if shuffle:
        np.random.shuffle(file_list)
    for f in tqdm(file_list, desc='Reading files'):
        mat = sio.loadmat(f)
        img, mask = np.array(mat['imgMat'],
                             dtype='float32'), np.array(mat['maskMat'],
                                                        dtype='int64')

        npy_data.extend(
            (img[:, :, i], mask[:, :, i]) for i in range(img.shape[2]))
    return npy_data


def read_mat_list_to_npy3d(file_list,
                           shuffle=False,
                           img_tag='imgMat',
                           mask_tag='maskMat'):
    npy_data = []
    if shuffle:
        np.random.shuffle(file_list)
    for f in tqdm(file_list, desc='Reading files'):
        mat = sio.loadmat(f)
        img, mask = np.array(mat[img_tag],
                             dtype='float32'), np.array(mat[mask_tag],
                                                        dtype='int64')
        npy_data.append((img, mask))
    return npy_data


def read_img_list_to_npy(file_list, color_mode, shuffle=False):

    def _read_image(image_file_name, mode=None):
        """
         Read image from file.
        :param image_file_name: full file path
        :param mode: 'gray', 'rgb' or 'idx'
        :return: numpy image
        """
        img = Image.open(image_file_name.rstrip())
        dty = 'float32'
        if mode is not None:
            if mode.lower() == 'gray':
                img = img.convert('L')
                dty = 'float32'
            else:
                if mode.lower() == 'rgb':
                    img = img.convert('RGB')
                    dty = 'float32'
                else:
                    if mode.lower() == 'idx':
                        img = img.convert('P')
                        dty = 'int64'
        return np.asarray(img, dtype=dty)

    npy_data = []
    l = len(file_list)
    if shuffle:
        np.random.shuffle(file_list)
    for f in tqdm(file_list, desc='Reading files'):
        im = _read_image(f, color_mode)
        npy_data.append(im)
    return npy_data


def split_dataset_npy(npy_file,
                      split_ratio=(0.8, 0.2, 0),
                      split_idx=None,
                      shuffle=True):
    """
    :param npy_file: 'data.npy'
    :param split_ratio: (train,test,valid) = (0.8, 0.2, 0)
    :param split_idx: example: split to two set train and test [[1,3,5,7,9],[0,2,4,6,8]]
    :param shuffle: True or False
    :return: train_set, test_set, valid_set, ...
    """
    data = np.load(npy_file, allow_pickle=True)
    split_data = []
    if not split_idx:
        if shuffle:
            np.random.shuffle(data)
        data_num = len(data)
        split_rg = [
            int(np.ceil(element * data_num)) for element in split_ratio
        ]
        s = sum(split_rg[:2])
        train = data[:split_rg[0]]
        test = data[split_rg[0]:s]
        valid = data[s:]
    else:
        for idx in split_idx:
            if shuffle:
                tmp = [data[i] for i in idx]
                np.random.shuffle(tmp)
                split_data.append(tmp)
            else:
                split_data.append([data[i] for i in idx])
        train, test, *valid = split_data
    return train, test, valid


def apply_colormap(im_gray):
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
    ])
    return colormap[im_gray]


def find_best_row_col(n):
    t = np.int(np.ceil(np.sqrt(n)))
    rg = np.arange(1, t + 1)
    cols, rows = 1, t
    for i in rg[::-1]:
        if n % i == 0:
            rows = i
            cols = np.int(n / rows)
            break
    cols, rows = [rows, cols] if cols < rows else [cols, rows]
    rows, cols = [np.int(np.ceil(n / t)), t
                  ] if rows == 1 or cols / rows > 3 else [rows, cols]
    return rows, cols


if __name__ == "__main__":
    path = 'F:\Data4LayerSegmentation\_Dataset_v2_'
    print(listFiles(path, '**/*.mat'))
