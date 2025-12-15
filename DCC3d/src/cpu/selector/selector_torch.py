from abc import ABC, abstractmethod
from enum import StrEnum
from typing import NotRequired, TypedDict

import torch


def calc_pairwise_distance(coords: torch.Tensor) -> torch.Tensor:
    r"""
    高效计算点云的全对全 (All-to-All) 欧氏距离矩阵。

    该函数利用了数学恒等式 $(a-b)^2 = a^2 + b^2 - 2ab$ 来将距离计算转化为矩阵乘法。
    相比于直接广播 (N, 1, 3) - (1, N, 3)，这种方法能显著降低临时显存/内存开销。

    Args:
        coords (torch.Tensor): 输入点云坐标。
            Shape: (N, 3) 或 (N, D)，其中 D 是特征维度。
            Dtype: float32 或 float64。

    Returns:
        torch.Tensor: 距离矩阵。
            Shape: (N, N)。矩阵中 (i, j) 位置的值表示第 i 个点和第 j 个点之间的欧氏距离。
            Dtype: 与输入保持一致。

    Mathematical Formulation:
        设输入矩阵为 X (N, D)。
        我们计算 $D_{ij} = ||x_i - x_j||_2$。
        展开后: $||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>$

        代码对应关系:
        - `square_sum`: $||x||^2$ (N, 1)
        - `torch.mm(coords, coords.T)`: $X \cdot X^T$ 即点积项 (N, N)

    Note:
        1. **数值稳定性**: 由于浮点数精度误差，$a^2 + b^2 - 2ab$ 可能会产生极小的负数（如 -1e-8）。
           因此必须使用 `torch.clamp(..., min=0)` 进行截断，否则 `torch.sqrt` 会产生 NaN。
        2. **对角线**: 理论上对角线元素应该严格为 0，但计算结果可能是极小的正数或 0。
    """
    square_sum = torch.sum(coords**2, dim=1, keepdim=True)
    dist_sq = square_sum + square_sum.T - 2 * torch.mm(coords, coords.T)
    dist_sq = torch.clamp(dist_sq, min=0)
    return torch.sqrt(dist_sq)


class BaseSelector(ABC):
    def __init__(self, n_sample: int):
        self.n_sample = n_sample

    @abstractmethod
    def select(self, coords: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        return self.select(coords)


class KNNSelector(BaseSelector):
    """
    基于欧氏距离的 K 近邻 (K-Nearest Neighbors) 选点策略。

    该策略为点云中的每一个点寻找几何距离最近的 k 个邻居。
    适用于点云密度相对均匀，且需要固定拓扑结构的场景。

    Attributes:
        n_sample (int): 每个中心点需要聚合的邻居数量 (k)。
    """

    def select(self, coords: torch.Tensor) -> torch.Tensor:
        """
        执行 KNN 选点计算。

        使用广播机制计算全对全 (All-to-All) 距离矩阵，并通过 topk
        快速提取前 k 个最近点的索引。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3), 其中 N 是点的数量，3 代表 (x, y, z)。
                Dtype: 通常为 float32 或 float64。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (N, n_sample).
                Dtype: int64 (整数索引).
                Values: 范围在 [0, N-1] 之间。

        Note:
            当前实现使用了 O(N^2) 的全局距离矩阵计算，对于 N > 10,000 的大规模点云，
            建议在生产环境中使用 KD-Tree 或 GPU 优化的 CUDA 算子。
        """
        dist_mat = calc_pairwise_distance(coords)
        _, knn_indices = torch.topk(dist_mat, self.n_sample, dim=1, largest=False)
        return knn_indices.long()


class BallQuerySelector(BaseSelector):
    """
    基于半径的球查询 (Ball Query / Radius Search) 选点策略。

    该策略以每个点为球心，寻找给定半径 R 内的所有点。
    适用于需要关注固定物理尺度特征的场景（如 PointNet++）。

    由于球内点数量不固定，该算法包含填充（Padding）和截断（Truncation）逻辑，
    以保证输出 Tensor 形状固定。

    Attributes:
        n_sample (int): 目标邻居数量 (k)。如果实际邻居不足，将重复采样；如果过多，将截断。
        radius (float): 搜索半径 (R)。
    """

    def __init__(self, n_sample: int, radius: float) -> None:
        """
        Args:
            n_sample (int): 每个点期望输出的邻居数。
            radius (float): 搜索球体的半径。
        """
        super().__init__(n_sample)
        self.radius = radius

    def select(self, coords: torch.Tensor) -> torch.Tensor:
        """
        执行 Ball Query 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (N, n_sample)。

        Implementation Details:
            1. 计算全对全距离矩阵。
            2. 筛选距离 < radius 的点。
            3. **截断策略**: 如果邻居数 > n_sample，取前 n_sample 个（通常是距离最近的，取决于排序稳定性）。
            4. **填充策略**: 如果邻居数 < n_sample：
               - 如果该点是孤立点（无邻居），则全部填充为自身索引。
               - 如果有部分邻居，使用第一个邻居的索引重复填充剩余位置。
        """
        N = coords.shape[0]
        device = coords.device
        dist_mat = calc_pairwise_distance(coords)
        indices = torch.zeros((N, self.n_sample), dtype=torch.long, device=device)

        for i in range(N):
            candidates = torch.where(dist_mat[i] < self.radius)[0]
            k = len(candidates)

            if k == 0:
                indices[i, :] = i
            elif k >= self.n_sample:
                valid_dist = dist_mat[i, candidates]
                sorted_local_idx = torch.argsort(valid_dist)
                indices[i, :] = candidates[sorted_local_idx[: self.n_sample]]
            else:
                indices[i, :k] = candidates
                indices[i, k:] = candidates[0]

        return indices


class DilatedKNNSelector(BaseSelector):
    """
    空洞 K 近邻 (Dilated KNN) 选点策略。

    类似于图像处理中的空洞卷积（Dilated Convolution）。
    该策略首先寻找最近的 (k * dilation) 个点，然后每隔 (dilation) 个点采样一次。

    目的：在不增加计算量（聚合点数 k 不变）的前提下，扩大感受野（Receptive Field）。

    Attributes:
        n_sample (int): 最终输出的邻居数量 (k)。
        dilation (int): 膨胀系数 (d)。
            - d=1: 等同于普通 KNN。
            - d=2: 每隔一个点取一个，感受野扩大约 2 倍。
    """

    def __init__(self, n_sample: int, dilation: int = 2) -> None:
        """
        Args:
            n_sample (int): 最终需要的邻居数。
            dilation (int, optional): 膨胀步长。默认为 2。
        """
        super().__init__(n_sample)
        self.dilation = dilation

    def select(self, coords: torch.Tensor) -> torch.Tensor:
        """
        执行 Dilated KNN 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (N, n_sample)。

        Logic:
            1. 搜索范围：计算最近的 (n_sample * dilation) 个邻居。
            2. 采样：使用 PyTorch 切片 `[::dilation]` 进行稀疏采样。
            3. 边界处理：如果实际找到的邻居总数不足以支持膨胀采样（例如点云边缘），
               将回退到普通 KNN 策略以保证输出形状正确。
        """
        N = coords.shape[0]
        dist_mat = calc_pairwise_distance(coords)
        sorted_indices = torch.argsort(dist_mat, dim=1)
        search_range = self.n_sample * self.dilation
        if search_range > N:
            raise ValueError(f"Dilated range {search_range} exceeds points {N}")
        dilated_indices = sorted_indices[:, 0 : search_range : self.dilation]
        if dilated_indices.shape[1] < self.n_sample:
            return sorted_indices[:, : self.n_sample].long()

        return dilated_indices.long()


class SelectorType(StrEnum):
    KNN = "knn"
    BALL_QUERY = "ball_query"
    DILATED = "dilated"


class SelectorConfig(TypedDict):
    type: SelectorType | str
    n: int
    radius: NotRequired[float]
    dilation: NotRequired[int]


class SelectorFactory:
    @staticmethod
    def get_selector(config: SelectorConfig) -> BaseSelector:
        try:
            raw_type = config["type"]
            if isinstance(raw_type, SelectorType):
                selector_type = raw_type
            else:
                selector_type = SelectorType(raw_type.lower())
        except ValueError:
            raise ValueError(f"Invalid selector type: {raw_type}")

        match selector_type:
            case SelectorType.KNN:
                return KNNSelector(n_sample=config["n"])
            case SelectorType.BALL_QUERY:
                if "radius" not in config:
                    raise ValueError("Config for BallQuery must contain 'radius'")
                return BallQuerySelector(n_sample=config["n"], radius=config["radius"])
            case SelectorType.DILATED:
                dilation = config.get("dilation", 2)
                return DilatedKNNSelector(n_sample=config["n"], dilation=dilation)
            case _:
                raise ValueError(f"Unknown selector type: {selector_type}")
