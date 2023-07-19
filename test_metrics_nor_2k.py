import importlib
import argparse
import cv2
import glob
import numpy as np
import os
import torch
import yaml
from copy import deepcopy
import random
from omegaconf import OmegaConf

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.data.paired_image_petrel_test_dataset import PairedImagePetrelTestDataset

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from basicsr.data import build_dataloader, build_dataset

from tqdm import tqdm
from collections import OrderedDict

from utils import util_net

metric_module = importlib.import_module('basicsr.metrics')


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


def get_dataloader():
    with open('configs/double_test.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg_data = cfg['datasets']['val']
    dataset = PairedImagePetrelTestDataset(cfg_data)
    loader = DataLoader(dataset, batch_size = 1, shuffle=False)
    # data = next(iter(loader))

    # img_in = data['in']
    # gt = data['gt']

    # # print(img_in.shape, img_in.mean())
    # # print('gt info: ', gt.shape, gt.mean())

    # img_in_data = tensor2img(img_in, rgb2bgr=True, out_type = np.float32)
    # img_in_data = img_in_data * 0.5 + 0.5

    # gt_data = tensor2img(gt, rgb2bgr=True, out_type = np.float32)
    # gt_data = gt_data * 0.5 + 0.5

    # cv2.imwrite('tmp/in.png', img_in_data)
    # cv2.imwrite('tmp/gt.png', gt_data)

    return loader


def setup_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_new_noise_schedule(model, schedule_opt, schedule_phase='train', device='cuda'):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.set_new_noise_schedule(schedule_opt, device=device)


def prepare_data(data):
    data = {key:value.cuda() for key, value in data.items()}
    return data


class Tester():
    def __init__(self, opt):
        with open('configs/double_test.yml', 'r') as f:
            cfg = yaml.safe_load(f)
        self.opt = cfg
        self.configs = opt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        launcher = 'slurm'
        init_dist(launcher)
        rank, world_size = get_dist_info()

        # datasets, dataloaders = {}, {}
        import models.networks as networks
        opt = self.configs.model.get('params', dict)
        # model = networks.define_G(opt).to(self.device)
        model = networks.define_G(opt)

        if world_size > 1:
            model = DDP(model.cuda(), device_ids=[rank,])  # wrap the network
        else:
            model = model.cuda()


        gen_path = '/mnt/lustre/sunjixiang1/code/DifFace/logs/06201639/ckpts/model_800000_gen.pth'
        # gen_path = '/mnt/lustre/sunjixiang1/code/DifFace/logs/06191112/ckpts/model_800000_gen.pth'

        ckpt = torch.load(gen_path, map_location=self.device)
        # print(ckpt['state_dict'].keys())
        # print(model.state_dict().keys())
        # set_new_noise_schedule(model, configs['model']['params']['beta_schedule']['val'], schedule_phase='val')
        if isinstance(model, nn.parallel.DistributedDataParallel):
            set_new_noise_schedule(model.module, self.configs['model']['params']['beta_schedule']['train'], schedule_phase='train', device=f"cuda:{rank}")
        else:
            set_new_noise_schedule(model, self.configs['model']['params']['beta_schedule']['train'], schedule_phase='train', device=f"cuda:{rank}")
        # set_new_noise_schedule(model, self.configs['model']['params']['beta_schedule']['train'], schedule_phase='train')

        util_net.reload_model(model, ckpt['state_dict'])
        # set_new_noise_schedule(model, configs['model']['params']['beta_schedule']['val'], schedule_phase='val')
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.set_ddim(self.configs['model']['params']['beta_schedule']['val']['n_timestep'])
            model.module.set_ddim(2000)
        else:
            model.set_ddim(self.configs['model']['params']['beta_schedule']['val']['n_timestep'])
            model.set_ddim(2000)

        torch.cuda.empty_cache()
        iters_start = ckpt['iters_start']
        setup_seed(iters_start)
        # cfg_vqgan = cfg['network_g']
        # model_type = cfg_vqgan.pop('type')
        # model = VQDoubleAutoEncoder(**cfg_vqgan)
        # model = VQDoubleFlipAutoEncoder(**cfg_vqgan)
        # model = VQSingleAutoEncoder(**cfg_vqgan)
        # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_164941_VQGAN-512-ds32-nearest-stage1/models/net_g_latest.pth'
        # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165038_VQGAN-512-ds32-nearest-stage1/models/net_g_latest.pth'
        # model_path = '/mnt/lustre/sunjixiang1/code/CodeFormer/experiments/20230619_165111_VQGAN-512-ds32-nearest-stage1/models/net_g_latest.pth'
        # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # for k, v in checkpoint['params_ema'].items():
        #     print(k)

        # model.load_state_dict(checkpoint['params_ema'])
        # model.to(self.device)
        self.net_g = model
        # model.eval()

    def feed_data(self, data):
        self.lq = (data['lq'] * 2 - 1).to(self.device)
        if 'gt' in data:
            self.gt = (data['gt'] * 2 - 1).to(self.device)
    
    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        # print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq

    # def nondist_validation(self, dataloader, current_iter, tb_logger,
                            # save_img, rgb2bgr, use_image):
    def val(self):
        rgb2bgr = True
        use_image = True
        dataloader = get_dataloader()
        # dataset_name = dataloader.dataset.opt['name']

        with open('configs/double_test.yml', 'r') as f:
            cfg = yaml.safe_load(f)
        with_metrics = cfg['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in cfg['val']['metrics'].keys()
            }
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        # pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue
            # img_name = os.path.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # if img_name[-1] != '9':
            #     continue

            # print('val_data .. ', val_data['lq'].size(), val_data['gt'].size())
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # if save_img:
                
            #     if self.opt['is_train']:
                    
            dir = 'tmp/val'
            save_img_path = os.path.join(dir, 'lq_2k_nor', os.path.basename(val_data['lq_path'][0]))
            save_gt_img_path = os.path.join(dir, 'gt_2k_nor', os.path.basename(val_data['gt_path'][0]))
            #     else:
                    
            #         save_img_path = osp.join(
            #             self.opt['path']['visualization'], dataset_name,
            #             f'{img_name}.png')
            #         save_gt_img_path = osp.join(
            #             self.opt['path']['visualization'], dataset_name,
            #             f'{img_name}_gt.png')
                    
            cv2.imwrite(save_img_path, sr_img)
            cv2.imwrite(save_gt_img_path, gt_img)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test')

            # pbar.update(1)
            # pbar.set_description(f'Test')
            # cnt += 1
            
            # curr_metric = {}
            # for metric in self.metric_results.keys():
            #     curr_metric[metric] = self.metric_results[metric] / cnt
            # print(cnt, curr_metric)
            # if cnt == 2:
            #     break
        if rank == 0:
            pbar.close()

        # pbar.close()
        
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():  # psnr // ssim
                # print('metric keys: ', metric)
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            # print('key name', name)
            keys.append(name)
            metrics.append(value)

        # print('metrics len: ', len(metrics))

        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)

        if rank == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt
            return metrics_dict




        # current_metric = 0.
        # if with_metrics:
        #     for metric in self.metric_results.keys():
        #         self.metric_results[metric] /= cnt
        #         current_metric = self.metric_results[metric]

        # if rank == 0:
        #     current_metric = 0.
        #     if with_metrics:
        #         for metric in self.metric_results.keys():
        #             self.metric_results[metric] /= cnt
        #             current_metric = self.metric_results[metric]
        # return current_metric, self.metric_results

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                # pred, _, _ = self.net_g(self.lq[i:j, :, :, :])
                if isinstance(self.net_g, nn.parallel.DistributedDataParallel):
                    pred = self.net_g.module.ddpm_sample(self.lq[i:j, :, :, :], False)
                else:
                    pred = self.net_g.ddpm_sample(self.lq[i:j, :, :, :], False)
                # pred = self.net_g(self.lq[i:j, :, :, :])
                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu() * 0.5 + 0.5
        out_dict['result'] = self.output.detach().cpu() * 0.5 + 0.5
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu() * 0.5 + 0.5
        return out_dict



if __name__ == '__main__':
    # with open('options/single_test.yml', 'r') as f:
    #     cfg = yaml.safe_load(f)
    cfg_path = '/mnt/lustre/sunjixiang1/code/DifFace/configs/training/degradation_test.yaml'

    cfg = OmegaConf.load(cfg_path)
    tester = Tester(cfg)
    # val_res, val_dict = tester.val()
    val_dict = tester.val()
    print(val_dict)
