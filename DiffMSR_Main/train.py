# flake8: noqa
import os
import os.path as osp
import sys
#
sys.path.append('/mnt/e/CVPR2024/DiffMSR/')
os.environ['RANK'] = str(0)
from DiffMSR_Main.train_pipeline import train_pipeline

# import DiffMSR_Main.archs
# import DiffMSR_Main.data
# import DiffMSR_Main.models
# import DiffMSR_Main.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
