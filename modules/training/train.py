"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys
import tqdm

def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--data_path', type=str, required=True,  
                        help='Path to the dataset JSON file.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')

    # args = parser.parse_args()

    print(args_list)

    # 如果 args_list 为 None，则解析命令行参数；否则解析 args_list
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args



import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import sys
import os

# 获取项目根目录路径（根据你的实际路径调整）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from modules.Xfeatmodel import *
from modules.training.utils import *
from modules.training.losses import *

from torch.utils.data import Dataset, DataLoader


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """

# 在 train.py 的顶部添加导入语句
from modules.dataset.dataload import CustomDataset
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, 
                 synthetic_root_path, 
                 ckpt_save_path, 
                 data_path,
                 model_name='xfeat_default',
                 batch_size=10, 
                 n_steps=160_000, 
                 lr=3e-4, 
                 gamma_steplr=0.5, 
                 training_res=(800, 608), 
                 device_num="0", 
                 dry_run=False,
                 save_ckpt_every=500):

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        # Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)


        # 初始化数据集和数据加载器
        self.dataset = CustomDataset(data_path)  # 替换为你的数据集路径
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.data_iter = iter(self.data_loader)

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name

    def train(self):
        self.net.train()

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                # 加载数据
                try:
                    image1, image2 = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    image1, image2 = next(self.data_iter)

                # 将数据移动到设备
                image1 = image1.to(self.dev)
                image2 = image2.to(self.dev)

                # 前向传播
                features1 = self.net(image1)
                features2 = self.net(image2)

                # 计算特征提取的损失函数
                # 假设 features1 和 features2 是特征提取器的输出
                loss = smooth_l1_loss(features1, features2)

                # 反向传播
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                # 记录日志
                pbar.set_description(f'Loss: {loss.item():.4f}')
                pbar.update(1)

                # 定期保存模型
                if (i + 1) % self.save_ckpt_every == 0:
                    checkpoint_path = os.path.join(self.ckpt_save_path, f'checkpoint_step_{i + 1}.pth')
                    torch.save({
                        'step': i + 1,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f'Checkpoint saved at step {i + 1} to {checkpoint_path}')

            # 训练结束后保存最终模型为 .pt 文件
            final_model_path = os.path.join(self.ckpt_save_path, 'xfeat_final_model.pt')
            self.save_model(final_model_path)

def run_training(data_path, ckpt_save_path, synthetic_root_path=None,  **kwargs):
    """
    封装训练逻辑，方便在其他文件中调用。
    :param data_path: 数据集路径
    :param ckpt_save_path: 检查点保存路径
    :param synthetic_root_path: 合成数据集路径（可选）
    :param kwargs: 其他训练参数
    """

    
    args_list = [
        '--synthetic_root_path', synthetic_root_path,
        '--ckpt_save_path', ckpt_save_path,
        '--data_path', data_path
        # 如果需要，还可以添加其他参数
        # '--training_type', 'xfeat_default',
        # ...
    ]

    print(args_list)
    args = parse_arguments(args_list)

    trainer = Trainer(
        synthetic_root_path=synthetic_root_path or args.synthetic_root_path,
        ckpt_save_path=ckpt_save_path,
        data_path=data_path,
        **kwargs
    )
    trainer.train()

if __name__ == '__main__':
    args = parse_arguments()

    run_training(
        data_path=args.data_path,
        ckpt_save_path=args.ckpt_save_path,
        synthetic_root_path=args.synthetic_root_path,
        megadepth_root_path=args.megadepth_root_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )
