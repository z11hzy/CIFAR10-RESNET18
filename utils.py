import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    固定随机种子，保证实验尽可能可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dirs():
    """
    创建实验所需文件夹。
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


class AverageMeter:
    """
    用于统计平均 loss 或 accuracy。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """
    计算 top-1 accuracy。
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        acc = correct / target.size(0) * 100.0
    return acc


def get_device():
    """
    自动选择 GPU / Apple MPS / CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")