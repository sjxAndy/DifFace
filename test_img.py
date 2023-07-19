#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os
import sys
import math
import time
import lpips
import random
import datetime
import functools
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

from datapipe.datasets import create_dataset
from models.resample import UniformSampler

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img

import utils.metrics as Metrics
import cv2


def to_tensor(img, is_train = False):
    '''
    convert numpy array to tensor
    for 2-D array [H , W] -> (1, 1, H , W)
    for 3-D array [H, W, 3] -> (1, 3, H, W)
    '''
    if is_train == False:
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 0)
        elif img.ndim == 3:
            if img.shape[2] == 3 or img.shape[2] == 2:  # for [H, W, 3] and [H, W, 2]
                img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)

        assert(img.ndim == 4)
        img = torch.from_numpy(img).float()

    else:
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        elif img.ndim == 3:
            if img.shape[2] == 3:  # for [H, W, 3] only
                img = np.transpose(img, (2, 0, 1))

        assert(img.ndim == 3)
        img = torch.from_numpy(img).float()

    return img

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        ndarray = tensor.cpu().detach().numpy()
        ndarray = np.squeeze(ndarray)

        if ndarray.shape[0] == 3:   # for [3, H, W] only
            ndarray = np.transpose(ndarray, (1, 2, 0))

        out = ndarray.copy()
    else:
        out = tensor

    return out

def setup_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_new_noise_schedule(model, schedule_opt, schedule_phase='train'):
    model.set_new_noise_schedule(schedule_opt, device=device)

def prepare_data(data):
    data = {key:value.cuda() for key, value in data.items()}
    return data

def get_img(gt_path):
    from petrel_client.client import Client
    from basicsr.utils import imfrombytes
    conf_path = '~/petreloss.conf'
    file_client = Client(conf_path)
    # load gt image
    # match with codebook ckpt data
    img_bytes = file_client.get(gt_path)
    return imfrombytes(img_bytes, float32=True)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cfg_path = '/mnt/lustre/sunjixiang1/code/DifFace/configs/training/degradation.yaml'

# configs = OmegaConf.load(cfg_path)

# datasets, dataloaders = {}, {}

# # dataset_config = configs.data.get('val', dict)
# # datasets['val'] = create_dataset(dataset_config)
# # dataloaders['val'] = udata.DataLoader(datasets['val'],
# #     batch_size=configs.train.batch[1],
# #     shuffle=False,
# #     drop_last=False,
# #     num_workers=0,
# #     pin_memory=True,
# #     )
# # print(len(datasets['val']))
# # val_data = datasets['val'][0]


# def read_img(pth = 'tmp/in0.png', pth1 = 'tmp/in1.png'):
#     # read img from
#     img = get_img('s3://Deblur/GoPro/crop/blur_crops/GOPR0868_11_01-000263_s006.png')
#     img_in = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)   
#     img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(pth, img_in*255)
#     img_in = to_tensor(img_in, is_train = False)

#     img_gt = get_img('s3://Deblur/GoPro/crop/sharp_crops/GOPR0868_11_01-000263_s006.png')
#     img_gt = cv2.resize(img_gt, (512, 512), interpolation = cv2.INTER_LINEAR).astype(np.float32)   
#     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(pth1, img_gt*255)
#     img_gt = to_tensor(img_gt, is_train = False)

#     return img_in, img_gt


# val_data = {}
# val_data['LQ'], val_data['HQ'] = read_img()
# print(val_data['LQ'].shape, val_data['LQ'].unsqueeze(0).shape, val_data['HQ'].shape)
# val_data = prepare_data(val_data)


# # print(type(val_data['LQ'].detach().float().cpu()))
# # lq_img = tensor2img(val_data['LQ'].detach().float().cpu(), rgb2bgr=True, out_type = np.float32)[0]
# # print(type(lq_img))
# # print(len(lq_img))
# # cv2.imwrite('tmp/LQnew.png', lq_img * 255)
# # print('ok')

# # for _,  val_data in enumerate(dataloaders['val']):
# #     print(_, val_data)

# import models.networks as networks
# opt = configs.model.get('params', dict)
# model = networks.define_G(opt).to(device)


# gen_path = '/mnt/lustre/sunjixiang1/code/DifFace/logs/06191112/ckpts/model_70000_gen.pth'
# opt_path = '/mnt/lustre/sunjixiang1/code/DifFace/logs/06191112/ckpts/model_70000_opt.pth'

# ckpt = torch.load(gen_path, map_location=device)
# # print(ckpt['state_dict'].keys())
# # print(model.state_dict().keys())
# set_new_noise_schedule(model, configs['model']['params']['beta_schedule']['val'], schedule_phase='val')

# util_net.reload_model(model, ckpt['state_dict'])
# torch.cuda.empty_cache()
# iters_start = ckpt['iters_start']
# setup_seed(iters_start)


# model.eval()
# with torch.no_grad():
#     HQ = model.super_resolution(val_data['LQ'], False)

# out_dict = OrderedDict()
# need_LR = True
# sample = False

# out_dict['HQ'] = HQ.detach().float().cpu()
# out_dict['LQ'] = val_data['LQ'].detach().float().cpu()
# out_dict['GT'] = val_data['HQ'].detach().float().cpu()
# visuals = out_dict
# # lq_img = Metrics.tensor2img(visuals['LQ'])  # uint8
# # hq_img = Metrics.tensor2img(visuals['HQ'])  # uint8
# # gt_img = Metrics.tensor2img(visuals['GT'])  # uint8

# lq_img = tensor2img(visuals['LQ'], rgb2bgr=True, out_type = np.float32)[0]
# hq_img = tensor2img(visuals['HQ'], rgb2bgr=True, out_type = np.float32)[0]
# gt_img = tensor2img(visuals['GT'], rgb2bgr=True, out_type = np.float32)[0]

# cv2.imwrite('tmp/HQ.png', hq_img * 255)
# cv2.imwrite('tmp/LQ.png', lq_img * 255)
# cv2.imwrite('tmp/GT.png', gt_img * 255)

img = cv2.imread('tmp/HQnew.png')
print(img.shape, img)
print(np.max(img), np.min(img))