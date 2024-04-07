import os
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
import scipy.io as sio
import numpy as np
from torchvision.transforms.functional import normalize
import torch
import random
import torch.fft as FFT

def FFT2D(x):
    return FFT.fftshift(FFT.fft2(x, dim=(-2, -1)))

def IFFT2D(x):
    return FFT.ifft2(FFT.ifftshift(x), dim=(-2, -1))

def to_tensor(data):
    return torch.from_numpy(data)

@DATASET_REGISTRY.register()
class MCSRPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MCSRPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.mask_path = opt['dataroot_mask']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        mask = sio.loadmat(self.mask_path)['lr_mask']
        mask = mask[np.newaxis, :, :]
        mask = np.concatenate([mask, mask], axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        gt_img = sio.loadmat(gt_path)

        gt_img_real = gt_img['T2'].real
        gt_img_real = gt_img_real[np.newaxis, :, :]
        gt_img_imag = gt_img['T2'].imag
        gt_img_imag = gt_img_imag[np.newaxis, :, :]

        # gt_img_real = (gt_img_real + 1) / 2
        # gt_img_imag = (gt_img_imag + 1) / 2

        gt_img = np.concatenate([gt_img_real, gt_img_imag], axis=0)  # 2,w,h
        img_gt = to_tensor(gt_img).float()
        # =======
        gt_k = img_gt.permute(1, 2, 0).contiguous()
        gt_k_ks = torch.view_as_real(FFT2D(torch.view_as_complex(gt_k))).contiguous()
        gt_ks = gt_k_ks.permute(2, 0, 1)
        

        lq_path = self.paths[index]['lq_path']
        lq_img = sio.loadmat(lq_path)

        lq_img_real = lq_img['T2_64'].real
        lq_img_real = lq_img_real[np.newaxis, :, :]
        lq_img_imag = lq_img['T2_64'].imag
        lq_img_imag = lq_img_imag[np.newaxis, :, :]

        # lq_img_real = (lq_img_real + 1) / 2
        # lq_img_imag = (lq_img_imag + 1) / 2

        lq_img = np.concatenate([lq_img_real, lq_img_imag], axis=0)
        img_lq = to_tensor(lq_img).float()

        ref_img = sio.loadmat(gt_path)

        ref_img_real = ref_img['T1'].real
        ref_img_real = ref_img_real[np.newaxis, :, :]
        ref_img_imag = ref_img['T1'].imag
        ref_img_imag = ref_img_imag[np.newaxis, :, :]

        # ref_img_real = (ref_img_real + 1) / 2
        # ref_img_imag = (ref_img_imag + 1) / 2

        ref_img = np.concatenate([ref_img_real, ref_img_imag], axis=0)

        img_ref = to_tensor(ref_img).float()

        return {'lq': img_lq, 'gt': img_gt, 'gt_k': gt_ks, 'ref': img_ref, 'mask': mask, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
