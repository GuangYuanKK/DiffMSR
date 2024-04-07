import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from DiffMSR_Main.models import lr_scheduler as lr_scheduler
from torch import nn
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
import torch.fft as FFT
import time

def FFT2D(x):
    return FFT.fftshift(FFT.fft2(x, dim=(-2, -1)))

def IFFT2D(x):
    return FFT.ifft2(FFT.ifftshift(x), dim=(-2, -1))

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space [b,w,h,2] need to [b,2,w,h]
    k0   - initially sampled elements in k-space
    dc_mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

def DC_layer(pred, gt_k, dc_mask):
    """
        pred - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
        zf   - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
        mask - [n*t,c,h,w] need to [n*t,h,w,c] [n,2,w,h] -> [n,w,h,2]
        """
    pred = pred.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]
    gt_k = gt_k.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]
    dc_mask = dc_mask.permute(0, 2, 3, 1).contiguous() # [n,w,h,2]


    #-----to compelx  [w,h]
    pred = torch.view_as_complex(pred) #[w,h]
    pred_k = FFT2D(pred)
    #-----to 2 channel
    pred_k = torch.view_as_real(pred_k)#[n,w,h,2]

    out_k = data_consistency(pred_k,gt_k,dc_mask) #[n,w,h,2]

    #-------to compelx
    out_k_ = torch.view_as_complex(out_k)
    out_dc = IFFT2D(out_k_) # compelx
    out_dc = torch.view_as_real(out_dc).contiguous()

    #------
    out_k = out_k.permute(0, 3, 1, 2).contiguous() # to [n,2,w,h]
    out_dc = out_dc.permute(0, 3, 1, 2).contiguous()  # to [n,2,w,h]

    return out_k, out_dc

@MODEL_REGISTRY.register()
class DiffMSRS2Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(DiffMSRS2Model, self).__init__(opt)
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)
        self.scale = self.opt.get('scale', 1)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            self.model_Es1 = self.net_g_S1.module.E
        else:
            self.model_Es1 = self.net_g_S1.E
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.encoder_iter = opt["train"]["encoder_iter"]
        self.lr_encoder = opt["train"]["lr_encoder"]
        self.lr_sr = opt["train"]["lr_sr"]
        self.gamma_encoder = opt["train"]["gamma_encoder"]
        self.gamma_sr = opt["train"]["gamma_sr"]
        self.lr_decay_encoder = opt["train"]["lr_decay_encoder"]
        self.lr_decay_sr = opt["train"]["lr_decay_sr"]



    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized in the second stage.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        parms=[]
        for k,v in self.net_g.named_parameters():
            if "denoise" in k or "condition" in k:
                parms.append(v)
        self.optimizer_e = self.get_optimizer(optim_type, parms, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_e)
        self.mse = nn.MSELoss()

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'LinearLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, train_opt['total_iter']))
        elif scheduler_type == 'VibrateLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, train_opt['total_iter']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.lq=data['lq'] .to(self.device)
        self.ref = data['ref'].to(self.device)
        self.gt_k = data['gt_k'].to(self.device)
        self.mask = data['mask'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(DiffMSRS2Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        ref = F.pad(self.ref, (0, mod_pad_w * scale, 0, mod_pad_h * scale), 'reflect')
        return lq,gt,ref,mod_pad_h,mod_pad_w

    def test(self):
        # print("testing!!!!!")
        # time_begin = time.time()
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq,gt,ref,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq
            gt=self.gt
            ref=self.ref
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq, ref, gt)
                self.out_k, self.out_dc = DC_layer(self.output, self.gt_k, self.mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq, ref, gt)
                self.out_k, self.out_dc = DC_layer(self.output, self.gt_k, self.mask)
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
            self.out_dc = self.out_dc[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        # time_end = time.time()
        # print("inference time: ", time_end-time_begin)

    def optimize_parameters(self, current_iter):
        # if current_iter < self.encoder_iter:
        #     lr_encoder = self.lr_encoder * (self.gamma_encoder ** ((current_iter ) // self.lr_decay_encoder))
        #     for param_group in self.optimizer_e.param_groups:
        #         param_group['lr'] = lr_encoder 
        # else:
        #     lr = self.lr_sr * (self.gamma_sr ** ((current_iter - self.encoder_iter ) // self.lr_decay_sr))
        #     for param_group in self.optimizer_g.param_groups:
        #         param_group['lr'] = lr 
        
        l_total = 0
        loss_dict = OrderedDict()
        _, S1_IPR = self.model_Es1(self.lq, self.gt)

        if current_iter < self.encoder_iter:
            self.optimizer_e.zero_grad()
            _, pred_IPR_list = self.net_g.module.diffusion(self.lq,S1_IPR[0])
            i=len(pred_IPR_list)-1
            S2_IPR=[pred_IPR_list[i]]
            l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
            l_total += l_abs
            loss_dict['l_kd_%d'%(i)] = l_kd
            loss_dict['l_abs_%d'%(i)] = l_abs

            l_total.backward()
            self.optimizer_e.step()
        else:
            self.optimizer_g.zero_grad()
            self.output, pred_IPR_list = self.net_g(self.lq, self.ref, S1_IPR[0])

            self.out_k, self.out_dc = DC_layer(self.output, self.gt_k, self.mask)

            l_pix = self.cri_pix(self.output, self.gt)

            k_dc = self.mse(self.out_k, self.gt_k) * 0.001

            l_total += l_pix
            l_total += k_dc
            loss_dict['l_pix'] = l_pix
            loss_dict['k_dc'] = k_dc

            i=len(pred_IPR_list)-1
            S2_IPR=[pred_IPR_list[i]]
            l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
            l_total += l_abs
            loss_dict['l_kd_%d'%(i)] = l_kd
            loss_dict['l_abs_%d'%(i)] = l_abs

            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)