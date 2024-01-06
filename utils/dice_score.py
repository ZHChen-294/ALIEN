import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

def BinaryDiceLoss(predict, target):
    smooth = 1
    p = 2
    reduction = 'mean'
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    dice = num / den

    loss = 1 - dice

    if reduction == 'mean':
        return loss.mean(), dice.mean()
    elif reduction == 'sum':
        return loss.sum(), dice.sum()
    elif reduction == 'none':
        return loss, dice
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.shape == target.shape, 'predict & target shape do not match'
    total_loss = 0

    dice_list = []
    for i in range(0, target.shape[1]):
        dice_loss, dice_value = BinaryDiceLoss(input[:, i], target[:, i])

        total_loss += dice_loss
        dice_list.append(dice_value.detach().cpu().numpy())

    return total_loss/(target.shape[1]-1), dice_list


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = SoftDiceLoss
    loss, dice = fn(input, target)
    return loss, dice


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    input = input.detach().cpu()
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, torch.as_tensor(input, dtype=torch.int64).cpu(), 1)

    return result




def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape
    y_onehot = gt

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


def SoftDiceLoss(input, target, batch_dice=True):
    smooth = 1
    loss_mask = None
    do_bg = False
    shp_x = input.shape

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    tp, fp, fn, _ = get_tp_fp_fn_tn(input, target, axes, loss_mask, False)

    nominator = 2 * tp + smooth
    denominator = 2 * tp + fp + fn + smooth

    dc = nominator / (denominator + 1e-8)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = torch.mean(dc[:, 1:], dim=0)

    return 1 - dc, dc.detach().cpu().numpy()
