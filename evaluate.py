import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DataParallel

from utils.dice_score import multiclass_dice_coeff, dice_coeff, make_one_hot, dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    if isinstance(net, DataParallel):
        n_classes = net.module.n_classes
    else:
        n_classes = net.n_classes

    net.eval()
    dice_score = []

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']

            image, mask_true = image.to(device=device, dtype=torch.float32), mask_true.to(device=device, dtype=torch.int)

            mask_pred = net(image)

            mask_true = torch.as_tensor(make_one_hot(mask_true.unsqueeze(dim=1), n_classes), device=mask_true.device)

            mask_pred = torch.as_tensor(make_one_hot(mask_pred.argmax(dim=1).unsqueeze(dim=1), n_classes), device=mask_pred.device)

            d_loss, d_value = dice_loss(mask_pred, mask_true)

            dice_score.append(d_value)

    net.train()
    return np.mean(dice_score, axis=0)
