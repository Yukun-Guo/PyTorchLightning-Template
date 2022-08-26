import torch
import numpy as np
import scipy.io as sio
import pytorch_lightning as ptl
from Module_NN import SegModel
from Backbone_NN import CNNNet
from torchmetrics import functional as MF
import torch.nn.functional as F

def read_mat_to_npy(filename):
    npy_data = []
    mat = sio.loadmat(filename)
    img, mask = np.array(mat['imgMat'], dtype='float32'), np.array(mat['maskMat'], dtype='int64')
    for i in range(img.shape[2]):
        npy_data.append((img[:, :, i], mask[:, :, i]))
    print('reading {}'.format(filename))
    return npy_data


def k_fold_split(file_list, fold=5):
    kFold_list_file = []
    kFold_list_idx = []
    data_set_length = len(file_list)
    idexs = np.tile(np.arange(0, fold), data_set_length // fold + 1)[:data_set_length]
    for i in range(fold):
        flg_valid = np.where(idexs == i)[0]
        flg_train = np.where(idexs != i)[0]
        train_lst = [file_list[j] for j in flg_valid]
        Valid_lst = [file_list[j] for j in flg_train]
        kFold_list_idx.append([flg_train, flg_valid])
        kFold_list_file.append([train_lst, Valid_lst])
    return kFold_list_file, kFold_list_idx


def read_file_list(list_txt_file):
    fp = open(list_txt_file, 'r')
    files = fp.readlines()
    files = [item.rstrip() for item in files]
    return files


def pad_power2_size(img, downsample_level=4):
    scale = np.power(2, downsample_level)
    r = np.ceil(img.shape[0] / scale) * scale
    c = np.ceil(img.shape[1] / scale) * scale
    pad_width = [[0, (r - img.shape[0]).astype(int)], [0, (c - img.shape[1]).astype(int)]]
    out_img = np.pad(img, pad_width)
    return out_img


def restore_size(img, raw_size):
    out_img = img[:raw_size[0], :raw_size[1]]
    return out_img


# load model
model_path = 'logs_newdata_norm-0-1_total_1/myBackboneNet/myBackboneNet-fold=1-epoch=015-val_loss=1.01050.ckpt'
model = SegModel(CNNNet, None, None).load_from_checkpoint(model_path)
model.freeze()
model.cuda()

# load data
file_list = r"F:\Data4LayerSegmentation\_Dataset\_ArrangedData\totalDataset\_mat_list.txt"
files = read_file_list(file_list)
kFold_list_list, kFold_list_idx = k_fold_split(files, fold=4)
print(kFold_list_list[0][0])
n_classes = 11
for fn in files:
    mat = sio.loadmat(fn)
    img, mask = np.array(mat['imgMat'], dtype='float32'), np.array(mat['maskMat'], dtype='int64')
    pred = np.zeros((n_classes, *img.shape)).astype('uint16')
    pred_int = np.zeros_like(img).astype('uint8')
    for i in range(img.shape[2]):
        # pre-pare data
        im = pad_power2_size(img[:, :, i], downsample_level=4)/255.0
        im = np.expand_dims(np.expand_dims(im, 0), 0)
        # predict
        y = model(torch.tensor(im).cuda())[1].cpu()
        y = F.softmax(y, dim=1, _stacklevel=5)
        # post-pare data
        out_int = torch.squeeze(torch.argmax(y, dim=1)).numpy()
        out_int = restore_size(out_int, img.shape[0:2]).astype('uint8')
        pred_int[:, :, i] = out_int

        out_dec = (torch.squeeze(y).numpy()*65535).astype('uint16')
        out_dec = out_dec[:, :img.shape[0], :img.shape[1]]
        pred[:, :, :, i] = out_dec
        print(i)
    sf = fn[0:-4]+'_pred.mat'
    sio.savemat(sf, {'pred': pred,'pred_int': pred_int})