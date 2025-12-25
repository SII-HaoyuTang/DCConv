import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DCC3d.src.cpu.dcconv3d_batchnrom import Dcconv3dBatchnorm
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from data.dataset import PointCloudQM9Dataset, PointCloudCollater, PointCloudTransform

def get_dataloader():
    # 数据集路径
    points_csv = "qm9.csv"
    indices_csv = "qm9_indices.csv"

    # 创建变换
    transform = PointCloudTransform(
        normalize_pos=False,
        center_pos=True,
        random_rotate=False,  # 训练时开启数据增强
    )

    # 创建完整数据集
    full_dataset = PointCloudQM9Dataset(
        points_csv=points_csv,
        indices_csv=indices_csv,
        transform=transform,
        target_column="internal_energy",
        node_features=[
            "atom_mass",
            "atom_valence_electrons",
            "atom_radius",
            "atom_mulliken_charge",
        ],  # 根据实际列名调整
    )

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建collator
    collater = PointCloudCollater(follow_batch=["pos", "x"])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class Dcconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Dcconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)