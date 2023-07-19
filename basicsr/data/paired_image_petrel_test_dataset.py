import os.path as osp
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from petrel_client.client import Client

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lq, img_gt


@DATASET_REGISTRY.register()
class PairedImagePetrelTestDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImagePetrelTestDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.gt_folder, self.lq_folder = opt['sdk_gt'], opt['sdk_lq']

        if self.file_client is None:
            conf_path = '~/petreloss.conf'
            self.file_client = Client(conf_path)
        self.text_url = self.opt['meta_info_file']
        # text_url = 's3://Deblur/GoPro/crop/blur_crops.lmdb/meta_info.txt'

        meta_txt = self.file_client.get(self.text_url)
        meta_txt = meta_txt.decode()

        gt_names = [meta_txt.split('\n')[i].split(' ')[0] for i in range(len(meta_txt.split('\n')))]

        self.paths = []
        for gt_name in gt_names:
            if gt_name.endswith('.jpg') or gt_name.endswith('.png'):
                folder, name = gt_name.split('-')
                pth = {}
                pth['gt_path'] = osp.join(self.gt_folder, folder, 'sharp', name)
                pth['lq_path'] = osp.join(self.lq_folder, folder, 'blur', name)
                self.paths.append(pth)



        # self.paths = []
        # contents = self.file_client.list(self.gt_folder)
        # for content in contents:
        #     if content.endswith('.jpg') or content.endswith('.png'):
        #         pth = {}
        #         pth['gt_path'] = osp.join(self.gt_folder, content)
        #         pth['lq_path'] = osp.join(self.lq_folder, content)
        #         self.paths.append(pth)
        #     else:
        #         continue

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

        # # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # padding
        #     img_gt, img_lq = padding(img_gt, img_lq, gt_size)
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # img_lq = img_lq * 2 - 1
        # img_gt = img_gt * 2 - 1

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
