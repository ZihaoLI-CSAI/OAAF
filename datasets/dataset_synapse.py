import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def downsample_label_preserve_small(label, out_h, out_w, rare=(8, 4), common=(1, 2, 3, 6, 7, 5)):
    """Block max-pool per class. Later rare classes overwrite earlier ones."""
    label = np.asarray(label, np.int64)
    h, w = label.shape
    if h == out_h and w == out_w:
        return label.copy()
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f'invalid low_res ({out_h}, {out_w})')
    if h % out_h != 0 or w % out_w != 0:
        return zoom(label, (out_h / h, out_w / w), order=0)
    bh, bw = h // out_h, w // out_w
    out = np.zeros((out_h, out_w), dtype=np.int64)
    rare = tuple(rare)
    common = tuple(common)
    for c in rare + common:
        if not np.any(label == c):
            continue
        mask = (label == c).astype(np.float32)
        pooled = mask.reshape(out_h, bh, out_w, bw).max(axis=(1, 3))
        active = pooled > 0.5
        if c in rare:
            out[active] = c
        else:
            out[active & (out == 0)] = c
    return out


def center_fit(image, label, out_h, out_w):
    h, w = image.shape
    out_image = np.zeros((out_h, out_w), np.float32)
    out_label = np.zeros((out_h, out_w), np.int64)
    src_y = max(0, (h - out_h) // 2)
    src_x = max(0, (w - out_w) // 2)
    dst_y = max(0, (out_h - h) // 2)
    dst_x = max(0, (out_w - w) // 2)
    copy_h = min(h, out_h)
    copy_w = min(w, out_w)
    out_image[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = image[src_y:src_y + copy_h, src_x:src_x + copy_w]
    out_label[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = label[src_y:src_y + copy_h, src_x:src_x + copy_w]
    return out_image, out_label


def random_scale(image, label, low=0.85, high=1.15):
    scale = random.uniform(low, high)
    h, w = image.shape
    scaled_h = max(1, int(round(h * scale)))
    scaled_w = max(1, int(round(w * scale)))
    image_s = zoom(image, (scaled_h / h, scaled_w / w), order=1)
    label_s = zoom(label, (scaled_h / h, scaled_w / w), order=0)
    return center_fit(image_s, np.rint(label_s).astype(np.int64), h, w)


def elastic_deform(image, label, alpha=30.0, sigma=10.0):
    h, w = image.shape
    dy = ndimage.gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    dx = ndimage.gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = [np.reshape(yy + dy, -1), np.reshape(xx + dx, -1)]
    image_w = ndimage.map_coordinates(image, coords, order=1, mode='reflect').reshape(h, w)
    label_w = ndimage.map_coordinates(label.astype(np.float32), coords, order=0, mode='constant', cval=0).reshape(h, w)
    return image_w.astype(np.float32), np.rint(label_w).astype(np.int64)


class RandomGenerator(object):
    def __init__(self, output_size, low_res, preserve_small_low_res=True,
                 rare_classes=None, common_classes=None, geom_aug=False):
        self.output_size = output_size
        self.low_res = low_res
        self.preserve_small_low_res = preserve_small_low_res
        self.rare_classes = rare_classes
        self.common_classes = common_classes
        self.geom_aug = geom_aug

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        if self.geom_aug and random.random() < 0.5:
            image, label = random_scale(image, label)
        if self.geom_aug and random.random() < 0.5:
            image, label = elastic_deform(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        if self.preserve_small_low_res:
            kwargs = {}
            if self.rare_classes is not None:
                kwargs['rare'] = self.rare_classes
            if self.common_classes is not None:
                kwargs['common'] = self.common_classes
            low_res_label = downsample_label_preserve_small(label, self.low_res[0], self.low_res[1], **kwargs)
        else:
            low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


def build_train_slice_sample_weights(base_dir, list_dir, split, small_classes=(4, 8), small_weight=4.0):
    """Phase C: higher DataLoader probability for slices containing small organs."""
    names = open(os.path.join(list_dir, split + '.txt')).readlines()
    weights = []
    n_small = 0
    for name in names:
        slice_name = name.strip('\n')
        data_path = os.path.join(base_dir, slice_name + '.npz')
        label = np.load(data_path)['label']
        has_small = any(np.any(label == c) for c in small_classes)
        if has_small:
            weights.append(float(small_weight))
            n_small += 1
        else:
            weights.append(1.0)
    stats = {'total': len(weights), 'small_organ_slices': n_small, 'small_weight': small_weight}
    return weights, stats


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
