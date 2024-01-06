import logging
import os.path

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd

from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom

def load_image(filename):
    img = sitk.ReadImage(filename)
    array = sitk.GetArrayFromImage(img)
    return img, array


def resize_3d(data, size, order=2, mode='reflect'):
    zoom_factors = np.asarray(size) / np.asarray(data.shape)
    data_r = zoom(data, zoom=zoom_factors, order=order, mode=mode)
    return data_r


class MedicalImageAugmentor:
    def __init__(self, config):
        self.config = config

    def __call__(self, img, label):
        self.config['img_fill_value'] = img.min()

        img_aug, label_aug = [img], [label]
        for i in range(self.config['aug_num']):
            img, label = self.random_rotate(img, label)
            img, label = self.random_zoom(img, label)
            img, label = self.random_crop_and_restore(img, label)
            img = self.add_gaussian_noise(img)
            img = self.add_gaussian_blur(img)
            img = self.adjust_brightness(img)
            img = self.adjust_contrast(img)
            img = self.augment_gamma(img)
            img, label = self.mirror(img, label)
            img, label = self.check_shape(img, label)
            assert img.shape == tuple(self.config['crop_shape']), 'The shape of img is wrong.'
            img_aug.append(img)
            label_aug.append(label)
        return img_aug, label_aug

    def random_rotate(self, img, label):
        assert img.shape == label.shape, 'The shape of img and label is different.'
        if np.random.uniform() <= self.config['p_rotate']:
            angle_x = np.random.uniform(-self.config['max_rotation'], self.config['max_rotation'])
            angle_y = np.random.uniform(-self.config['max_rotation'], self.config['max_rotation'])
            angle_z = np.random.uniform(-self.config['max_rotation'], self.config['max_rotation'])

            img = rotate(img, angle_x, axes=(1, 2), reshape=False, cval=self.config['img_fill_value'], order=3)
            img = rotate(img, angle_y, axes=(0, 2), reshape=False, cval=self.config['img_fill_value'], order=3)
            img = rotate(img, angle_z, axes=(0, 1), reshape=False, cval=self.config['img_fill_value'], order=3)

            label = rotate(label, angle_x, axes=(1, 2), reshape=False, cval=self.config['label_fill_value'], order=0)
            label = rotate(label, angle_y, axes=(0, 2), reshape=False, cval=self.config['label_fill_value'], order=0)
            label = rotate(label, angle_z, axes=(0, 1), reshape=False, cval=self.config['label_fill_value'], order=0)

        return img, label

    def random_zoom(self, img, label):
        assert img.shape == label.shape, 'The shape of img and label is different.'

        if np.random.uniform() <= self.config['p_scale']:
            scale = np.random.uniform(self.config['min_zoom_scale'], self.config['max_zoom_scale'])

            img = zoom(img, scale, order=3, mode='constant', cval=self.config['img_fill_value'])
            label = zoom(label, scale, order=0, mode='constant', cval=self.config['label_fill_value'])

        return img, label

    def random_crop_and_restore(self, img, label):
        assert img.shape == label.shape, 'The shape of img and label is different.'
        orig_shape = img.shape
        crop_shape = tuple(self.config['crop_shape'])

        if orig_shape[0] < crop_shape[0]:
            scale = np.asarray(crop_shape, dtype=np.float)/np.asarray(orig_shape, dtype=np.float)
            img = zoom(img, scale, order=3, mode='constant', cval=self.config['img_fill_value'])
            label = zoom(label, scale, order=0, mode='constant', cval=self.config['label_fill_value'])
        elif orig_shape[0] == crop_shape[0]:
            pass
        else:
            start = [np.random.randint(0, orig_shape[i] - crop_shape[i]) for i in range(3)]
            end = [start[i]+crop_shape[i] for i in range(len(start))]
            img = img[start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            label = label[start[0]: end[0], start[1]: end[1], start[2]: end[2]]

        return img, label

    def add_gaussian_noise(self, data):
        if np.random.uniform() <= self.config['p_gaussian_noise']:
            variance = np.random.uniform(0, self.config['gaussian_noise_variance'])
            noise = np.random.normal(0, variance, size=data.shape)
            data = data + noise
        return data

    def add_gaussian_blur(self, data):
        if np.random.uniform() <= self.config['p_gaussian_blur']:
            lower, upper = self.config['gaussian_blur_sigma_range'][0], self.config['gaussian_blur_sigma_range'][1]
            sigma = np.random.uniform(lower, upper)
            data = gaussian_filter(data, sigma=sigma, order=0)
        return data

    def adjust_brightness(self, data):
        if np.random.uniform() <= self.config['p_brightness']:
            alpha = np.random.uniform(1-self.config['max_brightness'], 1+self.config['max_brightness'])
            data = alpha * data
        return data

    def adjust_contrast(self, data):
        if np.random.uniform() <= self.config['p_contras']:
            factor = np.random.uniform(1-self.config['max_contrast'], 1+self.config['max_contrast'])
            mn, minm, maxm = data.mean(), data.min(), data.max()

            data = (data - mn) * factor + mn
            data = np.clip(data, a_min=minm, a_max=maxm)
        return data

    def augment_gamma(self, data, epsilon=1e-7):
        if np.random.uniform() <= self.config['p_gamma']:
            data_sample = - data
            mn, sd, minm = data_sample.mean(), data_sample.std(), data_sample.min()
            rnge = data_sample.max() - minm

            gamma = np.random.uniform(self.config['gamma_range'][0], self.config['gamma_range'][1])
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn

            data = - data_sample
        return data

    def mirror(self, data, label):
        if np.random.uniform() <= self.config['p_mirror']:
            if np.random.uniform() <= 0.5:
                data = data[::-1, :, :]
                label = label[::-1, :, :]

            if np.random.uniform() <= 0.5:
                data = data[:, ::-1, :]
                label = label[:, ::-1, :]

            if np.random.uniform() <= 0.5:
                data = data[:, :, ::-1]
                label = label[:, :, ::-1]

        return data, label

    def check_shape(self, data, label):
        orig_shape = data.shape
        crop_shape = tuple(self.config['crop_shape'])

        if orig_shape == crop_shape:
            pass
        else:
            scale = np.asarray(crop_shape) / np.asarray(orig_shape)
            data = zoom(data, scale, order=3, mode='constant', cval=self.config['img_fill_value'])
            label = zoom(label, scale, order=0, mode='constant', cval=self.config['label_fill_value'])
        return data, label


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, csv_file: str, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.csv_file = Path(csv_file)
        self.mask_suffix = mask_suffix

        file_content = pd.read_csv(csv_file, dtype=str)
        self.ids = np.array(file_content, dtype=str).reshape(-1)

        if self.ids.shape[0] == 0:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, is_mask):
        if not is_mask:
            img = np.expand_dims(img, axis=0)
        return img


    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = os.path.join(self.images_dir, str(name + '_0000' + self.mask_suffix))
        mask_file = os.path.join(self.mask_dir, str(name + self.mask_suffix))
        mask = sitk.ReadImage(mask_file)
        mask_config = {
            'name': str(name + self.mask_suffix),
            'spacing': mask.GetSpacing(),
            'origin': mask.GetOrigin(),
            'direction': mask.GetDirection()
        }

        img_s, img = load_image(img_file)
        mask_s, mask = load_image(mask_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)
        mask_config['shape'] = mask.shape

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'config': mask_config
        }

class TrainDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, csv_file):
        super().__init__(images_dir, mask_dir, csv_file, mask_suffix='.nii.gz')

        self.config = {
            "aug_num": 3,

            "max_rotation": 15,  #
            "max_brightness": 0.1,  #
            "max_contrast": 0.25,  ##
            "min_zoom_scale": 0.85,  ##
            "max_zoom_scale": 1.25,  ##
            "crop_shape": [48, 256, 256],  ##
            # "crop_shape": [96, 256, 256],  ##
            "gaussian_noise_variance": 0.01,  #
            "gaussian_blur_sigma_range": [0.5, 1],  # #
            "label_fill_value": 0,  #
            "gamma_range": [0.7, 1.5],  #

            "p_rotate": 0.2,
            "p_scale": 0.2,
            "p_gaussian_noise": 0.1,
            "p_gaussian_blur": 0.2,
            "p_brightness": 0.15,
            "p_contras": 0.15,
            "p_gamma": 0.3,
            "p_mirror": 0.5
        }
        self.augmenter = MedicalImageAugmentor(self.config)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = os.path.join(self.images_dir, str(name + '_0000' + self.mask_suffix))
        mask_file = os.path.join(self.mask_dir, str(name + self.mask_suffix))
        mask = sitk.ReadImage(mask_file)
        mask_config = {
            'name': str(name + self.mask_suffix),
            'spacing': mask.GetSpacing(),
            'origin': mask.GetOrigin(),
            'direction': mask.GetDirection()
        }

        img_s, img = load_image(img_file)
        mask_s, mask = load_image(mask_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'


        img_aug, mask_aug = self.augmenter(img, mask)


        img_aug = [torch.as_tensor(self.preprocess(i, is_mask=False).copy()).float().contiguous()
                   for i in img_aug]

        mask_aug = [torch.as_tensor(self.preprocess(i, is_mask=True).copy()).long().contiguous()
                    for i in mask_aug]

        return {
            'image': img_aug,
            'mask': mask_aug,
            'config': mask_config
        }


class TestDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, csv_file, scale=1):
        super().__init__(images_dir, mask_dir, csv_file, mask_suffix='.nii.gz')
