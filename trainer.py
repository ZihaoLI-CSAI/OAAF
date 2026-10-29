import argparse
import json
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, liver_gallbladder_prompts
from torchvision import transforms
from icecream import ic
from PIL import Image

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]


# def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8, ssl=False):
#     low_res_logits = outputs['low_res_logits']
#     # print(low_res_logits.size())
#     # print(low_res_label_batch.size())
#     loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
#     loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
#     loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
#     return loss, loss_ce, loss_dice
def build_ce_class_weights(num_classes, bg_weight, rare_classes, rare_weight, device):
    """CE weights: background down-weighted, automatically chosen rare classes up-weighted."""
    n = num_classes + 1
    w = torch.ones(n, dtype=torch.float32, device=device)
    w[0] = bg_weight
    for c in rare_classes:
        c = int(c)
        if 0 < c < n:
            w[c] = rare_weight
    return w


def contour_band(label, radius):
    """True on pixels within `radius` of a label boundary. label: [B,H,W]."""
    diff = torch.zeros_like(label, dtype=torch.bool)
    diff[:, 1:, :] |= label[:, 1:, :] != label[:, :-1, :]
    diff[:, :-1, :] |= label[:, :-1, :] != label[:, 1:, :]
    diff[:, :, 1:] |= label[:, :, 1:] != label[:, :, :-1]
    diff[:, :, :-1] |= label[:, :, :-1] != label[:, :, 1:]
    band = diff.float().unsqueeze(1)
    k = 2 * int(radius) + 1
    band = F.max_pool2d(band, kernel_size=k, stride=1, padding=int(radius))
    return band.squeeze(1) > 0.5


def boundary_ce_loss(logits, label, radius, ce_weight=None):
    """Cross-entropy restricted to a band around the full-resolution contour."""
    band = contour_band(label, radius)
    per_pixel = F.cross_entropy(logits, label.long(), weight=ce_weight, reduction='none')
    denom = band.sum().clamp_min(1).float()
    return (per_pixel * band.float()).sum() / denom


def build_dice_class_weights(num_classes, dice_fg_only):
    """Phase B4: optional zero weight on background channel in Dice."""
    n = num_classes + 1
    if not dice_fg_only:
        return None
    w = [0.0] + [1.0] * num_classes
    return w


def calc_loss(logits, label_batch, ce_loss, dice_loss, dice_weight: float = 0.8, dice_class_weights=None):
    loss_ce = ce_loss(logits, label_batch[:].long())
    loss_dice = dice_loss(logits, label_batch, softmax=True, weight=dice_class_weights)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def calc_train_loss(outputs, low_res_label_batch, full_res_label_batch, ce_loss, dice_loss,
                    dice_weight: float, lambda_hi: float, dice_class_weights=None,
                    boundary_weight: float = 0.0, boundary_radius: int = 2, ce_weight=None):
    loss_lo, loss_ce_lo, loss_dice_lo = calc_loss(
        outputs['low_res_logits'], low_res_label_batch, ce_loss, dice_loss, dice_weight, dice_class_weights)
    loss_hi = loss_ce_hi = loss_dice_hi = None
    if lambda_hi > 0:
        loss_hi, loss_ce_hi, loss_dice_hi = calc_loss(
            outputs['masks'], full_res_label_batch, ce_loss, dice_loss, dice_weight, dice_class_weights)
        loss = (1.0 - lambda_hi) * loss_lo + lambda_hi * loss_hi
    else:
        loss = loss_lo
    loss_boundary = None
    if boundary_weight > 0:
        loss_boundary = boundary_ce_loss(
            outputs['masks'], full_res_label_batch, boundary_radius, ce_weight)
        loss = loss + boundary_weight * loss_boundary
    return loss, loss_ce_lo, loss_dice_lo, loss_ce_hi, loss_dice_hi, loss_hi, loss_boundary

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
#椒盐噪声
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255 # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
        else:
            return img

def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    preserve_small = bool(getattr(args, 'preserve_small_low_res', 1))
    from datasets.dataset_synapse import scan_foreground_pixel_counts, select_rare_classes
    pixel_counts = scan_foreground_pixel_counts(
        args.root_path, args.list_dir, args.split, num_classes)
    manual_ids = []
    manual_text = str(getattr(args, 'rare_class_manual', '') or '').strip()
    if manual_text:
        manual_ids = [int(x) for x in manual_text.split(',') if x.strip()]
    rare_classes, overwrite_classes, fill_classes = select_rare_classes(
        pixel_counts, num_classes, int(getattr(args, 'rare_class_topk', 2)),
        mode=str(getattr(args, 'rare_class_mode', 'auto')), manual_ids=manual_ids)
    rare_stats = {
        'pixel_counts': {str(c): int(pixel_counts[c]) for c in range(num_classes + 1)},
        'rare_classes': [int(c) for c in rare_classes],
        'overwrite_order': [int(c) for c in overwrite_classes],
        'fill_order': [int(c) for c in fill_classes],
    }
    with open(os.path.join(snapshot_path, 'rare_class_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(rare_stats, f, indent=2)
    logging.info('Rare classes (fewest pixels first): %s', rare_classes)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=args.split,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                      low_res=[low_res, low_res],
                                                      preserve_small_low_res=preserve_small,
                                                      overwrite_classes=overwrite_classes,
                                                      fill_classes=fill_classes)
                                   ]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    use_oversample = bool(getattr(args, 'oversample_small_organs', 1))
    if use_oversample and args.split == 'train':
        from datasets.dataset_synapse import build_train_slice_sample_weights
        sample_weights, ow_stats = build_train_slice_sample_weights(
            args.root_path, args.list_dir, args.split,
            small_classes=rare_classes, small_weight=float(args.small_slice_weight))
        logging.info(
            'Oversample: %d / %d train slices contain rare classes %s; weight=%s',
            ow_stats['small_organ_slices'], ow_stats['total'], rare_classes, ow_stats['small_weight'])
        weight_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weight_tensor, num_samples=len(db_train), replacement=True)
        trainloader = DataLoader(
            db_train, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        trainloader = DataLoader(
            db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        # model = model.module
    model.train()
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    ce_device = torch.device('cuda')
    if getattr(args, 'use_class_weight_ce', 1):
        rare_ce = float(getattr(args, 'rare_ce_weight', args.ce_small_organ_weight))
        ce_w = build_ce_class_weights(
            num_classes, args.ce_bg_weight, rare_classes, rare_ce, ce_device)
        ce_loss = CrossEntropyLoss(weight=ce_w)
        logging.info('CE class weights (0-%d): %s', num_classes, ce_w.tolist())
    else:
        ce_loss = CrossEntropyLoss()
        ce_w = None
    cos_loss = CosineEmbeddingLoss()
    l1_loss = nn.L1Loss()
    dice_loss = DiceLoss(num_classes + 1)
    dice_class_weights = build_dice_class_weights(num_classes, bool(getattr(args, 'dice_fg_only', 1)))
    if dice_class_weights is not None:
        logging.info('Phase B4 Dice weights (0-8): %s', dice_class_weights)
    lambda_hi = float(getattr(args, 'lambda_hi', 0.45))
    boundary_weight = float(getattr(args, 'boundary_weight', 0.15))
    boundary_radius = int(getattr(args, 'boundary_radius', 2))
    use_gaussian = bool(getattr(args, 'use_gaussian_noise', 0))
    logging.info('lambda_hi (512 aux loss): %s', lambda_hi)
    logging.info('boundary loss weight=%s radius=%s', boundary_weight, boundary_radius)
    logging.info('gaussian noise: %s', use_gaussian)
    logging.info('preserve_small_low_res: %s', preserve_small)
    use_liver_prompt = bool(getattr(args, 'use_liver_prompt', 0))
    if use_liver_prompt and args.n_gpu > 1:
        raise RuntimeError('Liver prompt is implemented for n_gpu=1')
    if use_liver_prompt:
        logging.info(
            'Liver prompt: liver class %s -> target class %s, point_down=%s, box=%s, jitter=%s, '
            'box scale %s-%s, drop prob %s',
            args.prompt_liver_class, args.prompt_target_class, args.prompt_point_down,
            args.prompt_box_size, args.prompt_jitter, args.prompt_box_scale_min,
            args.prompt_box_scale_max, args.prompt_drop_prob)
    # dice_loss = Focal_loss(num_classes=num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            if use_gaussian:
                image_transform = image_batch.cpu()
                trans = transforms.Compose([
                    transforms.ToPILImage(),
                    AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 15), p=0.3),
                    transforms.ToTensor(),
                    ])
                img_list = []
                for i in range(image_transform.shape[0]):
                    img = image_transform[i]
                    img = trans(img)
                    img_list.append(img)
                image_transform = torch.stack(img_list, dim=0).cuda()
            else:
                image_transform = image_batch
            # outputs1,outputs2, attn1, attn2 = model(image_batch, multimask_output, args.img_size, gt=low_res_label_batch)
            # loss1, loss_ce1, loss_dice1 = calc_loss(outputs1, low_res_label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
            # loss2, loss_ce2, loss_dice2 = calc_loss(outputs2, label_batch, ce_loss, dice_loss, dice_weight=args.dice_param)
            # weight = 0.6**(0.990**epoch_num)
            # weight_self = 0.2
            # loss = (1-weight)*loss1 + (weight)*loss2
            outputs1, attn1 = model(image_transform, multimask_output, args.img_size, gt=label_batch)
            prompted = 0
            if use_liver_prompt and random.random() >= float(args.prompt_drop_prob):
                liver = label_batch == int(args.prompt_liver_class)
                coords, point_labels, boxes, valid = liver_gallbladder_prompts(
                    liver,
                    point_down=int(args.prompt_point_down),
                    box_size=int(args.prompt_box_size),
                    jitter=int(args.prompt_jitter),
                    scale_min=float(args.prompt_box_scale_min),
                    scale_max=float(args.prompt_box_scale_max))
                idx = valid.nonzero(as_tuple=False).view(-1)
                if idx.numel() > 0:
                    prompt_model = model.module if hasattr(model, 'module') else model
                    outputs2, _ = prompt_model.forward_prompt_from_cache(
                        idx, (coords[idx], point_labels[idx]), boxes[idx],
                        multimask_output, args.img_size, mode='train')
                    target = int(args.prompt_target_class)
                    masks = outputs1['masks'].clone()
                    low_res = outputs1['low_res_logits'].clone()
                    masks[idx, target] = outputs2['masks'][:, target]
                    low_res[idx, target] = outputs2['low_res_logits'][:, target]
                    outputs1 = {
                        'masks': masks,
                        'iou_predictions': outputs1['iou_predictions'],
                        'low_res_logits': low_res,
                    }
                    prompted = int(idx.numel())
            loss, loss_ce, loss_dice, loss_ce_hi, loss_dice_hi, loss_hi, loss_boundary = calc_train_loss(
                outputs1, low_res_label_batch, label_batch, ce_loss, dice_loss,
                dice_weight=args.dice_param, lambda_hi=lambda_hi, dice_class_weights=dice_class_weights,
                boundary_weight=boundary_weight, boundary_radius=boundary_radius, ce_weight=ce_w)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce_lo', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice_lo', loss_dice, iter_num)
            if loss_hi is not None:
                writer.add_scalar('info/loss_hi', loss_hi, iter_num)
                writer.add_scalar('info/loss_ce_hi', loss_ce_hi, iter_num)
                writer.add_scalar('info/loss_dice_hi', loss_dice_hi, iter_num)
            if loss_boundary is not None:
                writer.add_scalar('info/loss_boundary', loss_boundary, iter_num)
            # writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            # writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)
            # writer.add_scalar('info/loss_self2', loss_self, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item()))
            # logging.info('iteration %d : loss : %f, loss_ce1: %f, loss_dice1: %f, loss_ce2: %f, loss_dice2: %f' % (iter_num, loss.item(), loss_ce1.item(), loss_dice1.item(), loss_ce2.item(), loss_dice2.item()))
            boundary_val = loss_boundary.item() if loss_boundary is not None else 0.0
            prompt_msg = ''
            if use_liver_prompt:
                prompt_msg = ' | prompted %d/%d' % (prompted, int(image_batch.shape[0]))
            if loss_hi is not None:
                logging.info(
                    'iteration %d : loss %f (lo ce %f dice %f | hi ce %f dice %f | boundary %f)%s'
                    % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),
                       loss_ce_hi.item(), loss_dice_hi.item(), boundary_val, prompt_msg))
            else:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, boundary: %f%s'
                             % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), boundary_val, prompt_msg))



        save_interval = 25 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"