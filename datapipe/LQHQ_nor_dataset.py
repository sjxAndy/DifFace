from io import BytesIO
import lmdb
from PIL import Image
# from torch.utils.data import Dataset
import random
import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (
        adjust_brightness,
        adjust_contrast,
        adjust_hue,
        adjust_saturation,
        normalize
        )
from petrel_client.client import Client

from utils import util_common


@DATASET_REGISTRY.register()
class LQHQNorDataset(data.Dataset):
    """LQHQ Dataset
    """

    def __init__(self, opt):
        super(LQHQNorDataset, self).__init__()

        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder, self.lq_folder = opt['sdk_gt'], opt['sdk_lq']

        if self.file_client is None:
            conf_path = '~/petreloss.conf'
            self.file_client = Client(conf_path)
        self.text_url = self.opt['meta_info_file']
        # text_url = 's3://Deblur/GoPro/crop/blur_crops.lmdb/meta_info.txt'

        meta_txt = self.file_client.get(self.text_url)
        meta_txt = meta_txt.decode()
        # print(meta_txt)

        gt_names = [meta_txt.split('\n')[i].split(' ')[0] for i in range(len(meta_txt.split('\n')))]

        self.paths = []
        for gt_name in gt_names:
            if gt_name.endswith('.jpg') or gt_name.endswith('.png'):
                pth = {}
                pth['gt_path'] = osp.join(self.gt_folder, gt_name)
                pth['lq_path'] = osp.join(self.lq_folder, gt_name)
                self.paths.append(pth)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            conf_path = '~/petreloss.conf'
            self.file_client = Client(conf_path)
        
        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        img_lq = img_lq * 2 - 1
        img_gt = img_gt * 2 - 1

        return {'LQ': img_lq, 'HQ': img_gt, 'Index':index}
