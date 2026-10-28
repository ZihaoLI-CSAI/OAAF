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


def downsample_label_preserve_small(label, out_h, out_w, overwrite_classes, fill_classes):
    """Block max-pool. overwrite_classes replace overlaps; fill_classes only take empty cells."""
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
    overwrite_set = set(int(c) for c in overwrite_classes)
    for c in list(overwrite_classes) + list(fill_classes):
        c = int(c)
        if not np.any(label == c):
            continue
        mask = (label == c).astype(np.float32)
        pooled = mask.reshape(out_h, bh, out_w, bw).max(axis=(1, 3))
        active = pooled > 0.5
        if c in overwrite_set:
            out[active] = c
        else:
            out[active & (out == 0)] = c
    return out


class RandomGenerator(object):
    def __init__(self, output_size, low_res, preserve_small_low_res=True,
                 overwrite_classes=None, fill_classes=None):
        self.output_size = output_size
        self.low_res = low_res
        self.preserve_small_low_res = preserve_small_low_res
        self.overwrite_classes = list(overwrite_classes or [])
        self.fill_classes = list(fill_classes or [])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        if self.preserve_small_low_res:
            low_res_label = downsample_label_preserve_small(
                label, self.low_res[0], self.low_res[1],
                self.overwrite_classes, self.fill_classes)
        else:
            low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


def scan_foreground_pixel_counts(base_dir, list_dir, split, num_classes):
    """Count pixels of classes 0..num_classes over the split list."""
    names = open(os.path.join(list_dir, split + '.txt')).readlines()
    counts = np.zeros(num_classes + 1, dtype=np.int64)
    for name in names:
        slice_name = name.strip('\n')
        label = np.load(os.path.join(base_dir, slice_name + '.npz'))['label']
        ids, freq = np.unique(label, return_counts=True)
        for c, n in zip(ids.tolist(), freq.tolist()):
            c = int(c)
            if 0 <= c <= num_classes:
                counts[c] += int(n)
    return counts


def select_rare_classes(counts, num_classes, topk, mode='auto', manual_ids=None):
    """Rarest foreground classes by pixel count. overwrite order paints the rarest last."""
    ranked = sorted(
        [(c, int(counts[c])) for c in range(1, num_classes + 1)],
        key=lambda item: (item[1], item[0]))
    if mode == 'manual' and manual_ids:
        rare = [int(c) for c in manual_ids if 1 <= int(c) <= num_classes]
    else:
        rare = [c for c, _ in ranked[:max(int(topk), 0)]]
    rare_set = set(rare)
    fill = [c for c, _ in ranked if c not in rare_set]
    overwrite = list(reversed(rare))
    return rare, overwrite, fill


def build_train_slice_sample_weights(base_dir, list_dir, split, small_classes, small_weight=3.0):
    """Higher DataLoader probability for slices that contain a rare class."""
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
