import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from utils.dice_score import make_one_hot, dice_loss

from pathlib import Path
from utils.data_loading import TestDataset
from network import *
from torch.utils.data import DataLoader
import SimpleITK as sitk

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    with torch.no_grad():
        output = net(full_img).cpu()
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].squeeze().float().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output', '-o', metavar='OUTPUT', default='', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--trilinear', action='store_true', default=True, help='Use trilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    dir_img = Path('./data/Task_name/images')
    dir_mask = Path('/data/Task_name/labels')
    test_csv = Path('/data/Task_name/test.csv')
    model_path = Path('./checkpoints/Task_name/Model_name/Best_checkpoint_epoch.pth')
    out_path = Path('./output/Task_name/Model_name')
    save = True

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args.output = out_path

    dataset = TestDataset(dir_img, dir_mask, test_csv)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    n_data = len(dataset)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

    out_files = get_output_filenames(args)

    net = UNet_3D(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    args.model = model_path
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1, 2])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    dss = []
    for i, batch in enumerate(data_loader):
        image, true_mask = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.int)

        mask = predict_img(net=net,
                           full_img=image,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        mask_c = torch.as_tensor(np.expand_dims(mask, [0, 1]), device=true_mask.device)
        mask_c = torch.as_tensor(make_one_hot(mask_c, net.n_classes), device=mask_c.device)
        true_mask = torch.as_tensor(make_one_hot(true_mask.unsqueeze(dim=1), net.n_classes), device=true_mask.device)
        _, d = dice_loss(mask_c, true_mask)
        dss.append(d)

        if save:
            mask_f = sitk.GetImageFromArray(mask)
            config = batch['config']
            mask_f.SetSpacing(np.array([i.numpy() for i in config['spacing']]).reshape(-1))
            mask_f.SetOrigin(np.array([i.numpy() for i in config['origin']]).reshape(-1))
            mask_f.SetDirection(np.array([i.numpy() for i in config['direction']]).reshape(-1))
            out_filename = os.path.join(out_files, config['name'][0])
            sitk.WriteImage(mask_f, out_filename)
            logging.info(f'Mask saved to {out_filename}, Dice {d}')

    print(np.mean(dss, axis=0))
