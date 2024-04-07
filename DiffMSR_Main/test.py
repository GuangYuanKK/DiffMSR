# flake8: noqa
import os
import os.path as osp
import sys

sys.path.append('/mnt/e/CVPR2024/DiffMSR/')
os.environ['RANK'] = str(0)
import os.path as osp
from basicsr.test import test_pipeline

import DiffMSR_Main.archs
import DiffMSR_Main.data
import DiffMSR_Main.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
