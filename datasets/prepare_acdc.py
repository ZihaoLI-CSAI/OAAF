"""Turn official ACDC NIfTI volumes into the npz and h5 files used for training.

The patient lists follow BCP. Each patient's two annotated frames are renamed
frame01 and frame02 in increasing frame-number order. A volume is min-max
normalized, then every slice is resized to 256x256. Training slices are written
as npz files. Each test frame is written as one 3D npy.h5 volume.

Example:
    python datasets/prepare_acdc.py --src /path/to/ACDC/training --out ./data/ACDC
"""

import argparse
import os

import h5py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


def read_list(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]


def read_phases(cfg_path):
    ed = es = None
    with open(cfg_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip().upper()
            if key == 'ED':
                ed = int(value.strip())
            elif key == 'ES':
                es = int(value.strip())
    if ed is None or es is None:
        raise RuntimeError('ED or ES is missing from %s' % cfg_path)
    return ed, es


def find_patient_dir(src, patient):
    direct = [
        os.path.join(src, patient),
        os.path.join(src, 'training', patient),
        os.path.join(src, 'database', 'training', patient),
    ]
    for path in direct:
        if os.path.isfile(os.path.join(path, 'Info.cfg')):
            return path
    for root, dirs, _files in os.walk(src):
        if patient in dirs and os.path.isfile(os.path.join(root, patient, 'Info.cfg')):
            return os.path.join(root, patient)
    raise FileNotFoundError('Could not find %s under %s' % (patient, src))


def normalize(volume):
    volume = volume.astype(np.float32)
    low = float(volume.min())
    high = float(volume.max())
    if high > low:
        volume = (volume - low) / (high - low)
    else:
        volume = np.zeros_like(volume)
    return volume


def resize_slice(image, label, size):
    height, width = image.shape
    image = zoom(image, (size / height, size / width), order=3).astype(np.float32)
    label = zoom(label, (size / height, size / width), order=0).astype(np.int64)
    image = image[:size, :size]
    label = label[:size, :size]
    if image.shape != (size, size):
        canvas_image = np.zeros((size, size), np.float32)
        canvas_label = np.zeros((size, size), np.int64)
        canvas_image[:image.shape[0], :image.shape[1]] = image
        canvas_label[:label.shape[0], :label.shape[1]] = label
        image, label = canvas_image, canvas_label
    return image, label


def load_frame(patient_dir, patient, frame_number):
    stem = '%s_frame%02d' % (patient, frame_number)
    image_path = os.path.join(patient_dir, stem + '.nii.gz')
    label_path = os.path.join(patient_dir, stem + '_gt.nii.gz')
    if not os.path.isfile(image_path) or not os.path.isfile(label_path):
        raise FileNotFoundError('Missing %s or its label' % image_path)
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    if image.shape != label.shape:
        raise RuntimeError('Image and label shapes differ for %s' % stem)
    return normalize(image), label.astype(np.int64)


def resize_volume(image, label, size):
    slices = []
    labels = []
    for index in range(image.shape[0]):
        resized_image, resized_label = resize_slice(image[index], label[index], size)
        slices.append(resized_image)
        labels.append(resized_label)
    return np.stack(slices, axis=0), np.stack(labels, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Official ACDC training directory')
    parser.add_argument('--out', default='./data/ACDC', help='Output directory')
    parser.add_argument('--list_dir', default='./lists/lists_ACDC')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    train_names = read_list(os.path.join(args.list_dir, 'train.txt'))
    test_names = read_list(os.path.join(args.list_dir, 'test_vol.txt'))
    train_set = set(train_names)
    needed = {}
    for name in train_names:
        patient, rest = name.split('_', 1)
        frame = rest.split('_slice_')[0]
        needed.setdefault(patient, {}).setdefault(frame, set()).add(name)
    for name in test_names:
        patient, frame = name.split('_', 1)
        needed.setdefault(patient, {}).setdefault(frame, set())

    train_dir = os.path.join(args.out, 'train_npz')
    test_dir = os.path.join(args.out, 'test_vol_h5')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    written_slices = set()
    for patient, frames in sorted(needed.items()):
        patient_dir = find_patient_dir(args.src, patient)
        ed, es = read_phases(os.path.join(patient_dir, 'Info.cfg'))
        tags = {frame_number: tag for frame_number, tag in zip(sorted([ed, es]), ['frame01', 'frame02'])}
        for frame_number, tag in tags.items():
            if tag not in frames:
                continue
            image, label = load_frame(patient_dir, patient, frame_number)
            image, label = resize_volume(image, label, args.size)
            case = '%s_%s' % (patient, tag)
            if case in test_names:
                out_path = os.path.join(test_dir, case + '.npy.h5')
                with h5py.File(out_path, 'w') as handle:
                    handle.create_dataset('image', data=image, compression='gzip')
                    handle.create_dataset('label', data=label, compression='gzip')
            for index in range(image.shape[0]):
                slice_name = '%s_slice_%d' % (case, index)
                if slice_name not in train_set:
                    continue
                np.savez(
                    os.path.join(train_dir, slice_name + '.npz'),
                    image=image[index],
                    label=label[index],
                )
                written_slices.add(slice_name)
        print('converted %s' % patient)

    missing = sorted(train_set - written_slices)
    if missing:
        raise RuntimeError('Train slices were not created: %s' % ', '.join(missing[:10]))
    print('train slices %d, test volumes %d' % (len(written_slices), len(test_names)))


if __name__ == '__main__':
    main()
