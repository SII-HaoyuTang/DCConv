from abc import ABC, abstractmethod
from enum import StrEnum
from typing import NotRequired, TypedDict

import torch


def get_reduction_schedule(n_input, n_target, num_layers):
    """
    计算几何级数下降的调度表
    返回一个列表，包含每一层应该保留的点数
    """
    # 1. 确保输入是浮点数，否则 torch.log 会报错
    # 2. 在对数空间线性插值
    log_steps = torch.linspace(
        torch.log(torch.tensor(float(n_input))),
        torch.log(torch.tensor(float(n_target))),
        steps=num_layers + 1,
    )

    # 3. 转换回线性空间
    steps = torch.exp(log_steps)

    # 4. 取整并转换为长整型 (PyTorch 使用 .int() 或 .long())
    steps = torch.round(steps).long()

    # 5. 确保数值合法性
    # 使用 torch.clamp 确保最小值为 1
    steps = torch.clamp(steps, min=1)

    # 6. 返回第1层到第L层的目标点数 (去掉 steps[0])
    return steps[1:].tolist()


def monte_carlo_fill_tensor(
    positions, indices, target_n=30, margin=0.1, max_retries=200
):
    """
    使用带距离拒绝的蒙特卡洛采样填充点云 (Tensor版本)。

    Args:
        positions (torch.Tensor): (N, 3) 粒子的坐标
        indices (torch.Tensor): (N, 1) 或 (N,) 粒子所属的点云ID
        target_n (int): 目标点数
        margin (float): 边界框外扩的边距
        max_retries (int): 单次撒点的最大重试次数

    Returns:
        out_positions (torch.Tensor): (M * target_n, 3) 填充后的坐标
        out_indices (torch.Tensor): (M * target_n, 1) 对应的点云ID
    """
    # 确保 indices 是 1D 用于 mask，同时保持 device 一致
    device = positions.device
    if indices.dim() == 2:
        indices_flat = indices.squeeze(1)
    else:
        indices_flat = indices

    unique_groups = torch.unique(indices_flat)

    # 预分配结果列表
    result_pos_list = []
    result_idx_list = []

    for group_id in unique_groups:
        # 1. 获取当前组的坐标
        mask = indices_flat == group_id
        coords = positions[mask]  # (Current_N, 3)
        current_n = coords.shape[0]

        # Case A: 点数足够，进行下采样
        if current_n >= target_n:
            # 随机选择 target_n 个索引，无放回
            perm = torch.randperm(current_n, device=device)[:target_n]
            selected_coords = coords[perm]

            result_pos_list.append(selected_coords)
            result_idx_list.append(torch.full((target_n, 1), group_id, device=device))
            continue

        # Case B: 点数不足，需要蒙特卡洛填充
        # 2. 确定撒点范围 (Bounding Box)
        min_bound = torch.min(coords, dim=0)[0] - margin
        max_bound = torch.max(coords, dim=0)[0] + margin

        # 处理扁平/共线情况：如果某个轴范围极小，强制扩展
        diff = max_bound - min_bound
        # 找到范围过小的轴的掩码
        small_range_mask = diff < 1e-4
        if small_range_mask.any():
            center = (max_bound + min_bound) / 2
            # 强制给 0.5 的半径 (即 total width 1.0)
            min_bound[small_range_mask] = center[small_range_mask] - 0.5
            max_bound[small_range_mask] = center[small_range_mask] + 0.5

        # 3. 确定初始排斥半径 (Rejection Radius)
        if current_n > 1:
            # 计算两两距离矩阵 (Current_N, Current_N)
            pdist = torch.cdist(coords, coords)
            # 将对角线设为无穷大，避免自己和自己比较
            pdist.fill_diagonal_(float("inf"))
            # 取平均最近邻距离的 0.7 倍
            min_dist_threshold = torch.mean(torch.min(pdist, dim=1)[0]) * 0.7
        else:
            min_dist_threshold = torch.tensor(0.2, device=device)  # 默认值

        # 开始生成
        # 我们复制一份当前的坐标作为池子，因为新生成的点也会作为障碍物
        points_pool = coords.clone()
        points_needed = target_n - current_n

        # 临时存储生成的点
        new_points = []

        for _ in range(points_needed):
            success = False
            current_threshold = min_dist_threshold.clone()

            # 尝试 max_retries 次
            # 优化：为了避免Python循环过慢，我们可以一次生成一批候选点 (Batch Sampling)
            # 但为了严格遵守"生成一个，加入池子，影响下一个"的逻辑，这里保持逐个确认

            for retry in range(max_retries):
                # 在 Box 内随机生成一个点 (1, 3)
                candidate = (
                    torch.rand(1, 3, device=device) * (max_bound - min_bound)
                    + min_bound
                )

                # 计算到池子中所有点的距离
                # points_pool: (M, 3), candidate: (1, 3) -> dists: (1, M)
                dists = torch.cdist(candidate, points_pool)

                # 检查是否太近
                if torch.all(dists > current_threshold):
                    points_pool = torch.cat([points_pool, candidate], dim=0)
                    new_points.append(candidate)
                    success = True
                    break

            # 如果重试多次仍失败，说明太挤了，降低阈值并强制插入（或者重新尝试）
            # 原逻辑是：如果失败了，降低阈值继续在这个大循环里尝试生成这一个点
            # 但这里我们为了简化流程，若失败则降低阈值用于*下一个*点的生成判断，
            # 这里的实现稍微变通一下：如果当前点实在塞不进去，为了保证数目，
            # 我们就在当前最小距离允许的位置硬塞一个，或者再次尝试更小的阈值。
            if not success:
                # 策略：如果塞不进去，大幅降低阈值再试一次，或者直接盲选一个
                # 这里复刻原逻辑：缩小排斥半径 (注意：原逻辑是在外部while循环里缩小，这里模拟该行为)
                # 由于我们这里是固定次数循环，若失败，我们强制生成一个点但接受更小的距离
                candidate = (
                    torch.rand(1, 3, device=device) * (max_bound - min_bound)
                    + min_bound
                )
                points_pool = torch.cat([points_pool, candidate], dim=0)
                new_points.append(candidate)
                # 永久降低后续点的阈值
                min_dist_threshold *= 0.8

        # 拼接结果
        if len(new_points) > 0:
            generated_tensor = torch.cat(new_points, dim=0)
            full_coords = torch.cat([coords, generated_tensor], dim=0)
        else:
            full_coords = coords

        result_pos_list.append(full_coords)
        result_idx_list.append(torch.full((target_n, 1), group_id, device=device))

    # 最终合并
    out_positions = torch.cat(result_pos_list, dim=0)
    out_indices = torch.cat(result_idx_list, dim=0)

    return out_positions, out_indices


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
        self, coords: torch.Tensor, indices: torch.Tensor, n_sample: int, n_select: int
    ) -> torch.Tensor:
        """
        coords: (N, 3), indices: (N, 1)
        indices: the point cloud indices of each point
        n_sample: number of points to sample per point cloud
        n_select: number of points to select from the original N points
        Returns: (n_select, n_sample) tensor of selected neighbor indices
        """
        pass

    def __call__(
        self, coords: torch.Tensor, indices: torch.Tensor, n_sample: int, n_select: int
    ) -> torch.Tensor:
        return self.select(coords, indices, n_sample, n_select)


class KNNSelector(BaseSelector):
    """
    基于欧氏距离的 K 近邻 (K-Nearest Neighbors) 选点策略。

    该策略为点云中的每一个点寻找几何距离最近的 k 个邻居。
    适用于点云密度相对均匀，且需要固定拓扑结构的场景。

    Attributes:
        n_sample (int): 每个中心点需要聚合的邻居数量 (k)。
    """

    def select(
        self, coords: torch.Tensor, indices: torch.Tensor, n_sample: int, n_select: int
    ) -> torch.Tensor:
        """
        执行 KNN 选点计算。

        使用广播机制计算全对全 (All-to-All) 距离矩阵，并通过 topk
        快速提取前 k 个最近点的索引。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3), 其中 N 是点的数量，3 代表 (x, y, z)。
                Dtype: 通常为 float32 或 float64。
            indices (torch.Tensor): 输入点云索引矩阵。
                Shape: (N, 1) 或 (N,)。
            n_sample (int): 每个点需要选择的邻居数量。
            n_select (int): 从原始N个点中选择多少个点作为中心点。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (n_select, n_sample).
                Dtype: int64 (整数索引).
                Values: 范围在 [0, N-1] 之间。

        Note:
            当前实现使用了 O(N^2) 的全局距离矩阵计算，对于 N > 10,000 的大规模点云，
            建议在生产环境中使用 KD-Tree 或 GPU 优化的 CUDA 算子。
        """
        N = coords.shape[0]

        # First, select n_select points from the original N points using minimal variance strategy
        if n_select >= N:
            # If n_select >= N, use all points
            selected_centers_coords = coords
            center_indices = torch.arange(N, device=coords.device)
        else:
            # Use minimal variance strategy to select n_select points
            selected_centers_coords, center_indices = select_n_points_minimal_variance(
                coords, n_select
            )

        # Calculate distance matrix between selected centers and all points
        dist_mat = torch.cdist(selected_centers_coords, coords)

        # Find KNN for each selected center
        _, knn_indices = torch.topk(dist_mat, min(n_sample, N), dim=1, largest=False)

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

    def __init__(self, radius: float) -> None:
        """
        Args:
            radius (float): 搜索球体的半径。
        """
        super().__init__()
        self.radius = radius

    def select(
        self, coords: torch.Tensor, indices: torch.Tensor, n_sample: int, n_select: int
    ) -> torch.Tensor:
        """
        执行 Ball Query 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。
            indices (torch.Tensor): 输入点云索引矩阵。
                Shape: (N, 1) 或 (N,)。
            n_sample (int): 每个点需要选择的邻居数量。
            n_select (int): 从原始N个点中选择多少个点作为中心点。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (n_select, n_sample)。

        Implementation Details:
            1. 首先选择 n_select 个中心点
            2. 计算距离矩阵，筛选距离 < radius 的点。
            3. **截断策略**: 如果邻居数 > n_sample，取前 n_sample 个（通常是距离最近的，取决于排序稳定性）。
            4. **填充策略**: 如果邻居数 < n_sample：
               - 如果该点是孤立点（无邻居），则全部填充为自身索引。
               - 如果有部分邻居，使用第一个邻居的索引重复填充剩余位置。
        """
        N = coords.shape[0]
        device = coords.device

        # First, select n_select points from the original N points using minimal variance strategy
        if n_select >= N:
            # If n_select >= N, use all points
            selected_centers_coords = coords
            center_indices = torch.arange(N, device=coords.device)
        else:
            # Use minimal variance strategy to select n_select points
            selected_centers_coords, center_indices = select_n_points_minimal_variance(
                coords, n_select
            )

        # Calculate distance matrix between selected centers and all points
        dist_mat = torch.cdist(selected_centers_coords, coords)
        result_indices = torch.zeros(
            (n_select, n_sample), dtype=torch.long, device=device
        )

        for i in range(n_select):
            candidates = torch.where(dist_mat[i] < self.radius)[0]
            k = len(candidates)

            if k == 0:
                result_indices[i, :] = center_indices[i]
            elif k >= n_sample:
                valid_dist = dist_mat[i, candidates]
                sorted_local_idx = torch.argsort(valid_dist)
                result_indices[i, :] = candidates[sorted_local_idx[:n_sample]]
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
        n_sample (int): 最终输出的邻居数量 (k)。
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
        self, coords: torch.Tensor, indices: torch.Tensor, n_sample: int, n_select: int
    ) -> torch.Tensor:
        """
        执行 Dilated KNN 选点计算。

        Args:
            coords (torch.Tensor): 输入点云坐标矩阵。
                Shape: (N, 3)。
            indices (torch.Tensor): 输入点云索引矩阵。
                Shape: (N, 1) 或 (N,)。
            n_sample (int): 每个点需要选择的邻居数量。
            n_select (int): 从原始N个点中选择多少个点作为中心点。

        Returns:
            torch.Tensor: 选中的邻居索引矩阵。
                Shape: (n_select, n_sample)。

        Logic:
            1. 首先选择 n_select 个中心点
            2. 搜索范围：计算最近的 (n_sample * dilation) 个邻居。
            3. 采样：使用 PyTorch 切片 `[::dilation]` 进行稀疏采样。
            4. 边界处理：如果实际找到的邻居总数不足以支持膨胀采样（例如点云边缘），
               将回退到普通 KNN 策略以保证输出形状正确。
        """
        N = coords.shape[0]

        # First, select n_select points from the original N points using minimal variance strategy
        if n_select >= N:
            # If n_select >= N, use all points
            selected_centers_coords = coords
            center_indices = torch.arange(N, device=coords.device)
        else:
            # Use minimal variance strategy to select n_select points
            selected_centers_coords, center_indices = select_n_points_minimal_variance(
                coords, n_select
            )

        # Calculate distance matrix between selected centers and all points
        dist_mat = torch.cdist(selected_centers_coords, coords)
        sorted_indices = torch.argsort(dist_mat, dim=1)

        search_range = n_sample * self.dilation
        if search_range > N:
            # If search range exceeds total points, fall back to regular KNN
            return sorted_indices[:, :n_sample].long()

        dilated_indices = sorted_indices[:, 0 : search_range : self.dilation]
        if dilated_indices.shape[1] < n_sample:
            return sorted_indices[:, :n_sample].long()

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


default_config = SelectorConfig(type=SelectorType.KNN, n=16)


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
