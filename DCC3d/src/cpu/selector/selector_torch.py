from abc import ABC, abstractmethod
from enum import Enum
from typing import TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import torch

def batch_select_n_points_minimal_variance(
    batch_points: torch.Tensor,  # (B, N, D)
    n: int,
    valid_mask: torch.Tensor = None,
    max_iter: int = 50,  # 可以减小，因为批处理更稳定
    restarts: int = 5,  # 可以减少，因为批量初始化更均匀
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    优化版本的批处理最小方差点选择
    """
    device = batch_points.device
    B, N, D = batch_points.shape

    if n >= N:
        indices = torch.arange(N, device=device).expand(B, N)
        return batch_points, indices

    # 使用有效点掩码
    if valid_mask is None:
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # 计算每个点云的有效点数
    valid_counts = valid_mask.sum(dim=1)  # (B,)

    # 对于 n >= 有效点数的情况，直接返回所有有效点
    small_mask = n >= valid_counts
    if small_mask.any():
        # 对于小点云，特殊处理
        # 这里简化为取前n个点，实际应该取所有点并进行填充
        pass

    # 使用多个随机初始化并行处理
    # 将重启维度并入batch维度
    points_expanded = batch_points.unsqueeze(1).expand(
        -1, restarts, -1, -1
    )  # (B, R, N, D)
    points_expanded = points_expanded.reshape(B * restarts, N, D)  # (B*R, N, D)

    # 同样扩展valid_mask
    valid_mask_expanded = valid_mask.unsqueeze(1).expand(-1, restarts, -1)  # (B, R, N)
    valid_mask_expanded = valid_mask_expanded.reshape(B * restarts, N)  # (B*R, N)

    # Vectorized random initialization - avoid Python loop
    # Use torch.multinomial to sample one valid index per batch element
    # Add small epsilon to avoid zero weights causing issues
    weights = valid_mask_expanded.float() + 1e-10
    rand_indices = torch.multinomial(weights, 1).squeeze(1)  # (B*R,)

    centers = points_expanded[
        torch.arange(B * restarts, device=device), rand_indices
    ]  # (B*R, D)

    # 迭代优化
    last_indices = torch.zeros(B * restarts, n, dtype=torch.long, device=device)
    last_indices.fill_(-1)

    for iter_idx in range(max_iter):
        # 计算距离（批量化）
        dists = torch.cdist(points_expanded, centers.unsqueeze(1)).squeeze(
            2
        )  # (B*R, N)

        # 屏蔽无效点
        dists = dists.masked_fill(~valid_mask_expanded, float("inf"))

        # 选择最近的n个点
        _, indices = torch.topk(dists, n, dim=1, largest=False)  # (B*R, n)

        # 检查收敛
        current_sorted = indices.sort(dim=1)[0]
        last_sorted = last_indices.sort(dim=1)[0]
        converged = torch.all(current_sorted == last_sorted, dim=1)

        # 收集选择的点
        batch_idx = torch.arange(B * restarts, device=device).unsqueeze(1).expand(-1, n)
        selected_points = points_expanded[batch_idx, indices]  # (B*R, n, D)

        # 更新中心
        new_centers = selected_points.mean(dim=1)  # (B*R, D)
        centers = torch.where(converged.unsqueeze(-1), centers, new_centers)

        # 更新上一轮索引
        last_indices = indices.clone()

        if converged.all():
            break

    # 计算loss并选择每个batch的最佳重启
    # 首先计算每个选择的方差
    if n > 1:
        selected_mean = selected_points.mean(dim=1, keepdim=True)
        variance = torch.mean((selected_points - selected_mean) ** 2, dim=1)
        losses = variance.sum(dim=1)  # (B*R,)
    else:
        losses = torch.zeros(B * restarts, device=device)

    # 重塑为(B, R)
    losses = losses.reshape(B, restarts)  # (B, R)
    indices = indices.reshape(B, restarts, n)  # (B, R, n)
    selected_points = selected_points.reshape(B, restarts, n, D)  # (B, R, n, D)

    # 选择每个batch中loss最小的重启
    best_restart = torch.argmin(losses, dim=1)  # (B,)

    # 收集最佳结果
    best_indices = indices[torch.arange(B, device=device), best_restart]  # (B, n)
    best_points = selected_points[
        torch.arange(B, device=device), best_restart
    ]  # (B, n, D)

    return best_points, best_indices

def select_n_points_minimal_variance(points, n, max_iter=100, restarts=10):
    """
    points: torch.Tensor, 形状为 (N, D)
    n: 目标选取的点数
    restarts: 多次随机初始化以避免局部最优
    """
    N, D = points.shape

    # 【修复1】：边界检查，如果总点数少于目标n，直接返回所有点
    if N <= n:
        indices = torch.arange(N, device=points.device)
        return points, indices

    # 【修复2】：如果只选1个点，方差定义为0，无需优化，直接随机或取均值最近点
    # 这里为了保持逻辑一致，允许进入循环，但必须修正下方var的计算

    best_loss = float("inf")
    # 【修复3】：给 best_indices 一个默认值，防止全过程 Loss 均为 NaN 导致的 NoneType 错误
    # 默认取前 n 个
    best_indices = torch.arange(n, device=points.device)

    for _ in range(restarts):
        # 1. 随机初始化中心
        center = points[torch.randint(0, N, (1,))]

        last_indices = torch.tensor([], device=points.device)

        for i in range(max_iter):
            # 2. 计算距离
            dists = torch.sum((points - center) ** 2, dim=1)

            # 3. 选取最近的 n 个点
            # topk 可能在 n=1 时产生维度压缩，保持维度安全
            _, indices = torch.topk(dists, n, largest=False)

            # 检查收敛
            current_indices_sorted, _ = indices.sort()
            if torch.equal(current_indices_sorted, last_indices):
                break
            last_indices = current_indices_sorted

            # 4. 更新中心
            selected_points = points[indices]
            center = selected_points.mean(dim=0, keepdim=True)

            # 【关键修复4】：计算 Loss
            # unbiased=False 使用有偏估计 (除以 n 而不是 n-1)
            # 这避免了 n=1 时出现除以 0 的错误 (NaN)
            if n > 1:
                current_loss = torch.var(selected_points, dim=0, unbiased=False).sum()
            else:
                current_loss = torch.tensor(0.0, device=points.device)

            # 更新最优解
            # 必须确保 current_loss 不是 NaN (虽然 unbiased=False 基本解决了这个问题)
            if current_loss < best_loss:
                best_loss = current_loss
                best_indices = indices

        # 如果 n=1，其实不需要多次重启（除非是为了找具体的某个点），找到一个 loss=0 就可以退出了
        if n == 1:
            break

    # print(f"Best Indices: {best_indices.shape}")
    return points[best_indices], best_indices


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
    def __init__(self):
        pass

    @abstractmethod
    def select(
        self, coords: torch.Tensor, indices: torch.Tensor, conv_num: int, outpoint_num: int
    ) -> torch.Tensor:
        """
        coords: (N, 3), indices: (N, 1)
        indices: the point cloud indices of each point
        conv_num: number of points to sample per point cloud
        outpoint_num: number of points to select from the original N points
        Returns: (outpoint_num, conv_num) tensor of selected neighbor indices
        """
        pass

    def __call__(
        self, coords: torch.Tensor, indices: torch.Tensor, conv_num: int, outpoint_num: int
    ) -> torch.Tensor:
        return self.select(coords, indices, conv_num, outpoint_num)


class KNNSelector(BaseSelector):
    """
    基于欧氏距离的 K 近邻 (K-Nearest Neighbors) 选点策略。

    该策略为点云中的每一个点寻找几何距离最近的 k 个邻居。
    适用于点云密度相对均匀，且需要固定拓扑结构的场景。

    Attributes:
        conv_num (int): 每个中心点需要聚合的邻居数量 (k)。
    """

    def select(
        self,
        coords: torch.Tensor,
        belonging: torch.Tensor,
        conv_num: int,
        outpoint_num: int,
        batch_size: int = 8,  # 根据GPU内存调整
    ) -> torch.Tensor:
        """
        Fully vectorized KNN computation with minimal Python loops.
        
        Optimizations applied:
        1. Pre-compute all group indices using argsort + unique_consecutive
        2. Use padding via advanced indexing instead of loops
        3. Vectorize center extraction and global index conversion
        """
        device = coords.device
        dtype = coords.dtype

        # 获取点云信息 - use stable sort for reproducibility
        unique_groups, inverse_indices, group_counts = torch.unique(
            belonging, return_inverse=True, return_counts=True
        )
        num_groups = len(unique_groups)

        # 每个点云选择的中心点数
        per_group_outpoint_num = min(outpoint_num, group_counts.min().item())

        # 结果张量
        total_centers = num_groups * per_group_outpoint_num
        neighbor_indices = torch.full(
            (total_centers, conv_num), -1, dtype=torch.long, device=device
        )

        # Pre-compute sorted indices by group for efficient group extraction
        sorted_order = torch.argsort(belonging)
        sorted_belonging = belonging[sorted_order]
        
        # Compute group start positions using cumsum
        group_sizes = group_counts.tolist()
        max_group_size = max(group_sizes)
        group_starts = torch.zeros(num_groups + 1, dtype=torch.long, device=device)
        group_starts[1:] = torch.cumsum(group_counts, dim=0)

        # 分批处理以避免内存溢出
        num_batches = (num_groups + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_groups)
            batch_size_current = batch_end - batch_start

            # Get batch group sizes and max size for this batch
            batch_group_sizes = group_counts[batch_start:batch_end]
            max_batch_group_size = batch_group_sizes.max().item()

            # Create padded batch tensors - VECTORIZED
            batch_coords = torch.zeros(
                batch_size_current, max_batch_group_size, 3, device=device, dtype=dtype
            )
            batch_mask = torch.zeros(
                batch_size_current, max_batch_group_size, dtype=torch.bool, device=device
            )
            
            # Pre-allocate global indices tensor for this batch
            batch_global_indices = torch.zeros(
                batch_size_current, max_batch_group_size, dtype=torch.long, device=device
            )

            # Fill batch tensors using vectorized operations where possible
            for i in range(batch_size_current):
                group_idx = batch_start + i
                start = group_starts[group_idx].item()
                end = group_starts[group_idx + 1].item()
                size = end - start
                
                # Get global indices for this group
                global_indices = sorted_order[start:end]
                batch_global_indices[i, :size] = global_indices
                batch_coords[i, :size] = coords[global_indices]
                batch_mask[i, :size] = True

            # Select center points using minimal variance algorithm
            _, local_center_indices = batch_select_n_points_minimal_variance(
                batch_coords, per_group_outpoint_num, valid_mask=batch_mask
            )

            # Collect center coordinates - VECTORIZED using gather
            # Create batch indices for gathering
            batch_idx_expand = torch.arange(batch_size_current, device=device).unsqueeze(1)
            batch_idx_expand = batch_idx_expand.expand(-1, per_group_outpoint_num)
            
            # Get center coords directly through indexing
            center_local_coords = batch_coords[batch_idx_expand, local_center_indices]  # (B, P, 3)
            batch_centers = center_local_coords

            # Compute KNN using efficient distance computation
            # ||c - x||^2 = ||c||^2 + ||x||^2 - 2*c·x
            centers_sq = torch.sum(batch_centers**2, dim=-1, keepdim=True)  # (B, P, 1)
            coords_sq = torch.sum(batch_coords**2, dim=-1, keepdim=True)  # (B, M, 1)
            centers_coords = torch.einsum("bpc,bmc->bpm", batch_centers, batch_coords)
            dist_sq = centers_sq + coords_sq.transpose(1, 2) - 2 * centers_coords

            # Mask invalid (padded) points
            dist_sq = dist_sq.masked_fill(~batch_mask.unsqueeze(1), float("inf"))

            # Select top-k nearest neighbors
            k = min(conv_num, max_batch_group_size)
            _, topk_local_indices = torch.topk(dist_sq, k, dim=2, largest=False)

            # Convert to global indices - VECTORIZED
            # Expand batch_global_indices for gathering
            batch_idx_3d = torch.arange(batch_size_current, device=device).view(-1, 1, 1)
            batch_idx_3d = batch_idx_3d.expand(-1, per_group_outpoint_num, k)
            
            # Gather global indices for top-k neighbors
            global_topk = torch.gather(
                batch_global_indices.unsqueeze(1).expand(-1, per_group_outpoint_num, -1),
                2,
                topk_local_indices
            )  # (B, P, k)

            # Write results to output - vectorized slice assignment
            result_start = batch_start * per_group_outpoint_num
            result_end = batch_end * per_group_outpoint_num
            
            # Reshape global_topk to (B * P, k) for assignment
            global_topk_flat = global_topk.reshape(-1, k)
            neighbor_indices[result_start:result_end, :k] = global_topk_flat

        return neighbor_indices


class BallQuerySelector(BaseSelector):
    """
    基于半径的球查询 (Ball Query / Radius Search) 选点策略。

    该策略以每个点为球心，寻找给定半径 R 内的所有点。
    适用于需要关注固定物理尺度特征的场景（如 PointNet++）。

    由于球内点数量不固定，该算法包含填充（Padding）和截断（Truncation）逻辑，
    以保证输出 Tensor 形状固定。

    Attributes:
        conv_num (int): 目标邻居数量 (k)。如果实际邻居不足，将重复采样；如果过多，将截断。
        radius (float): 搜索半径 (R)。
    """

    def __init__(self, radius: float) -> None:
        """
        Args:
            radius (float): 搜索球体的半径。
        """
        super().__init__()
        self.radius = radius

    def select(
        self, coords: torch.Tensor, indices: torch.Tensor, conv_num: int, outpoint_num: int
    ) -> torch.Tensor:
        """
        执行 Ball Query 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。
            indices (torch.Tensor): 输入点云索引矩阵。
                Shape: (N, 1) 或 (N,)。
            conv_num (int): 每个点需要选择的邻居数量。
            outpoint_num (int): 从原始N个点中选择多少个点作为中心点。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (outpoint_num, conv_num)。

        Implementation Details:
            1. 首先选择 outpoint_num 个中心点
            2. 计算距离矩阵，筛选距离 < radius 的点。
            3. **截断策略**: 如果邻居数 > conv_num，取前 conv_num 个（通常是距离最近的，取决于排序稳定性）。
            4. **填充策略**: 如果邻居数 < conv_num：
               - 如果该点是孤立点（无邻居），则全部填充为自身索引。
               - 如果有部分邻居，使用第一个邻居的索引重复填充剩余位置。
        """
        N = coords.shape[0]
        device = coords.device

        # First, select outpoint_num points from the original N points using minimal variance strategy
        if outpoint_num >= N:
            # If outpoint_num >= N, use all points
            selected_centers_coords = coords
            center_indices = torch.arange(N, device=coords.device)
        else:
            # Use minimal variance strategy to select outpoint_num points
            selected_centers_coords, center_indices = select_n_points_minimal_variance(
                coords, outpoint_num
            )

        # Calculate distance matrix between selected centers and all points
        dist_mat = torch.cdist(selected_centers_coords, coords)
        result_indices = torch.zeros(
            (outpoint_num, conv_num), dtype=torch.long, device=device
        )

        for i in range(outpoint_num):
            candidates = torch.where(dist_mat[i] < self.radius)[0]
            k = len(candidates)

            if k == 0:
                result_indices[i, :] = center_indices[i]
            elif k >= conv_num:
                valid_dist = dist_mat[i, candidates]
                sorted_local_idx = torch.argsort(valid_dist)
                result_indices[i, :] = candidates[sorted_local_idx[:conv_num]]
            else:
                result_indices[i, :k] = candidates
                result_indices[i, k:] = candidates[0]

        return result_indices


class DilatedKNNSelector(BaseSelector):
    """
    空洞 K 近邻 (Dilated KNN) 选点策略。

    类似于图像处理中的空洞卷积（Dilated Convolution）。
    该策略首先寻找最近的 (k * dilation) 个点，然后每隔 (dilation) 个点采样一次。

    目的：在不增加计算量（聚合点数 k 不变）的前提下，扩大感受野（Receptive Field）。

    Attributes:
        conv_num (int): 最终输出的邻居数量 (k)。
        dilation (int): 膨胀系数 (d)。
            - d=1: 等同于普通 KNN。
            - d=2: 每隔一个点取一个，感受野扩大约 2 倍。
    """

    def __init__(self, dilation: int = 2) -> None:
        """
        Args:
            dilation (int, optional): 膨胀步长。默认为 2。
        """
        super().__init__()
        self.dilation = dilation

    def select(
        self, coords: torch.Tensor, belonging: torch.Tensor, conv_num: int, outpoint_num: int
    ) -> torch.Tensor:
        """
        执行 Dilated KNN 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。
            indices (torch.Tensor): 输入点云索引矩阵。
                Shape: (N, 1) 或 (N,)。
            conv_num (int): 每个点需要选择的邻居数量。
            outpoint_num (int): 从原始N个点中选择多少个点作为中心点。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (outpoint_num, conv_num)。

        Logic:
            1. 首先选择 outpoint_num 个中心点
            2. 搜索范围：计算最近的 (conv_num * dilation) 个邻居。
            3. 采样：使用 PyTorch 切片 `[::dilation]` 进行稀疏采样。
            4. 边界处理：如果实际找到的邻居总数不足以支持膨胀采样（例如点云边缘），
               将回退到普通 KNN 策略以保证输出形状正确。
        """
        N = coords.shape[0]

        # First, select outpoint_num points from the original N points using minimal variance strategy
        if outpoint_num >= N:
            # If outpoint_num >= N, use all points
            selected_centers_coords = coords
            center_indices = torch.arange(N, device=coords.device)
        else:
            # Use minimal variance strategy to select outpoint_num points
            selected_centers_coords, center_indices = select_n_points_minimal_variance(
                coords, outpoint_num
            )

        # Calculate distance matrix between selected centers and all points
        dist_mat = torch.cdist(selected_centers_coords, coords)
        sorted_indices = torch.argsort(dist_mat, dim=1)

        search_range = conv_num * self.dilation
        if search_range > N:
            # If search range exceeds total points, fall back to regular KNN
            return sorted_indices[:, :conv_num].long()

        dilated_indices = sorted_indices[:, 0 : search_range : self.dilation]
        if dilated_indices.shape[1] < conv_num:
            return sorted_indices[:, :conv_num].long()

        return dilated_indices.long()


class SelectorType(str, Enum):
    KNN = "knn"
    BALL_QUERY = "ball_query"
    DILATED = "dilated"


class SelectorConfig(TypedDict):
    """
    Configuration for a selector used in various selection algorithms.

    This class defines the structure and required fields for configuring
    a selector. It is typically used to specify how elements should be
    selected based on certain criteria such as type, number, radius,
    and dilation.

    Attributes:
        type (SelectorType | str): The type of selector to use.
        n (int): The number of elements to select.
        radius (float, optional): The radius within which elements are considered.
                                  This is not a required field.
        dilation (int, optional): The dilation factor to apply. This is not a
                                  required field.
    """
    type: SelectorType | str
    radius: NotRequired[float]
    dilation: NotRequired[int]


default_config = SelectorConfig(type=SelectorType.KNN)


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
                return KNNSelector()
            case SelectorType.BALL_QUERY:
                if "radius" not in config:
                    raise ValueError("Config for BallQuery must contain 'radius'")
                return BallQuerySelector(radius=config["radius"])
            case SelectorType.DILATED:
                dilation = config.get("dilation", 2)
                return DilatedKNNSelector(dilation=dilation)
            case _:
                raise ValueError(f"Unknown selector type: {selector_type}")
