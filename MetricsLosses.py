import torch
import torch.nn.functional as F
import torchmetrics as tm

# losses

def _jaccard_loss(y_pred, y_true, smooth=1., axis=(1, 2, 3)):
    intersection = torch.sum(y_true * y_pred, dim=axis)
    union = torch.sum(y_true + y_pred, dim=axis)
    loss = 1 - (intersection + smooth) / (union - intersection + smooth)
    return torch.mean(loss)


def my_jaccard_loss(y_pred, y_true, num_classes=3):
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes)
    y_onehot = torch.permute(y_true_onehot, [0, 3, 1, 2])
    bk_loss = _jaccard_loss(
        y_onehot[:, 0, :, :], y_pred[:, 0, :, :], axis=(1, 2))
    np_loss = _jaccard_loss(
        y_onehot[:, 1, :, :], y_pred[:, 1, :, :], axis=(1, 2))
    shd_loss = _jaccard_loss(
        y_onehot[:, 2, :, :], y_pred[:, 2, :, :], axis=(1, 2))
    return bk_loss * 0.1 + np_loss * 0.8 + shd_loss * 0.1


# metrics
def _jaccard_coef(y_true, y_pred, smooth=1.):
    # as known as IOU
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f + y_pred_f)
    return (intersection + smooth) / (union - intersection + smooth)


def my_jaccard_coef(y_true, y_pred, num_classes=3):
    # as known as IOU
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes)
    y_onehot = torch.permute(y_true_onehot, [0, 3, 1, 2])
    bk_jc = _jaccard_coef(y_onehot[:, 0, :, :], y_pred[:, 0, :, :])
    np_jc = _jaccard_coef(y_onehot[:, 1, :, :], y_pred[:, 1, :, :])
    shd_jc = _jaccard_coef(y_onehot[:, 2, :, :], y_pred[:, 2, :, :])

    return bk_jc, np_jc, shd_jc


if __name__ == '__main__':
    y_true = torch.tensor([2, 1])
    y_onehot = F.one_hot(y_true, num_classes=3)
    y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.5, 0.2]])
    j = _jaccard_loss(y_onehot, y_pred, axis=(1))
    y_s = F.softmax(y_pred, _stacklevel=5)
    ce = F.cross_entropy(y_pred, y_true)
    print(ce)
