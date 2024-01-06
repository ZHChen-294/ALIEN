import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
import time
from evaluate import evaluate
from network import *
from utils.data_loading import BasicDataset, TrainDataset, TestDataset
from utils.dice_score import dice_loss, make_one_hot
import numpy as np
from torch.nn.parallel import DataParallel

dir_img = Path('./data/Task_name/images')
dir_mask = Path('/data/Task_name/labels')
train_csv = Path('/data/Task_name/train.csv')
val_csv = Path('/data/Task_name/val.csv')
test_csv = Path('/data/Task_name/test.csv')
dir_checkpoint = Path('./checkpoints/Task_name/Model_name')

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-7,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
):
    if isinstance(model, DataParallel):
        n_classes = model.module.n_classes
    else:
        n_classes = model.n_classes

    # 1. Create dataset
    train_set = TrainDataset(dir_img, dir_mask, train_csv)
    val_set = TestDataset(dir_img, dir_mask, val_csv)
    test_set = TestDataset(dir_img, dir_mask, test_csv)

    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)

    # 2. Create dataloaders
    Train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    Val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    Test_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **Train_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **Val_loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **Test_loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Testing size: {n_test}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 4. Training
    iter_num, iter, max_iter = 1, 0, 250  # Just need to modify the maximum number of iterations (max_iter) for each training round.
    max_iter = max_iter / batch_size
    best_sc = 0
    best_epoch = 0
    early_stopping, patience = 0, 100  # Early stop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        stop_now = False
        model.train()
        epoch_loss = []
        dice_value = []
        for batch in train_loader:
            images_l, true_masks_l = batch['image'], batch['mask']
            for images, true_masks in zip(images_l, true_masks_l):
                iter = iter + 1
                images, true_masks = images.to(device=device, dtype=torch.float32), true_masks.to(device=device, dtype=torch.int)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)


                    ce_loss = criterion(masks_pred, torch.as_tensor(true_masks, dtype=torch.long))
                    d_loss, d_value = dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        torch.as_tensor(make_one_hot(true_masks.unsqueeze(dim=1), n_classes),
                                        device=true_masks.device)
                    )
                    loss = 1 * ce_loss + 1 * d_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                epoch_loss.append(loss.item())
                dice_value.append(d_value)

                if iter >= max_iter:
                    logging.info('Iter_num: {}, dice (train): {}, loss (epoch mean): {}'.format(iter_num, np.mean(dice_value, axis=0), np.mean(epoch_loss)))
                    iter = 0

                    val_score = evaluate(model, val_loader, device, amp)
                    logging.info('Validation Dice score: {}'.format(val_score))

                    if iter_num % 50 == 0:
                        state_dict = model.state_dict()
                        torch.save(state_dict, os.path.join(dir_checkpoint, 'Checkpoint_'+str(iter_num)+'_epoch.pth'))
                        logging.info(f'Checkpoint {iter_num} saved!')

                    if val_score[0] > best_sc:
                        best_epoch = iter_num
                        best_sc = val_score[0]
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        torch.save(state_dict, str(dir_checkpoint / 'Best_checkpoint_epoch.pth'))
                        logging.info(f'Best checkpoint {iter_num} saved!')
                    else:
                        early_stopping += 1

                    if early_stopping > patience:
                        stop_now = True
                        print(f"Early stopping at epoch {iter_num}...")
                        break

                    end_time = time.time()
                    logging.info('This epoch took {:.2f}s\n'.format(end_time - start_time))
                    start_time = time.time()

                    iter_num += 1
                    scheduler.step()

            if stop_now:
                break
        if stop_now:
            break

    best_state_dict = torch.load(str(dir_checkpoint / 'Best_checkpoint_epoch.pth'))
    model.load_state_dict(best_state_dict)
    test_loss, test_score = evaluate(model, test_loader, device, amp)
    logging.info('\nFinal dice (test): {}, loss (mean): {}'.format(test_score, test_loss[0]))
    logging.info('Best epoch is epoch {}, and the total iter num is {}'.format(best_epoch, global_step))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('--trilinear', action='store_true', default=True, help='Use trilinear upsampling')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # model = UNet_3D(n_channels=1, n_classes=args.classes, trilinear=args.trilinear)
    model = ALIEN(n_channels=1, n_classes=args.classes, trilinear=args.trilinear)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Trilinear" if model.trilinear else "Transposed conv"} upscaling')

    model.to(device=device)
    model = DataParallel(model, device_ids=[0])

    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        amp=args.amp
    )
