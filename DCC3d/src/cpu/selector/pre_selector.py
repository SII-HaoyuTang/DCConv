import torch


class PreSelector:
    def __init__(
        self,
        target_points: int,
        convlayer_nums: int,
        min_points=10,
        margin=0.8,
        max_retries=200,
    ):
        """
        A class for managing configuration parameters related to a specific operation, likely in a machine learning
        or data processing context.

        Parameters:
            target_points (int): The desired number of points to achieve.
            convlayer_nums (int): The number of convolutional layers to be used.
            min_points (int, optional): The minimum number of points required. Defaults to 10.
            margin (float, optional): A margin value, possibly used for tolerance or thresholding. Defaults to 0.8.
            max_retries (int, optional): The maximum number of attempts to retry an operation. Defaults to 200.
        """
        self.target_points = target_points
        self.convlayer_nums = convlayer_nums
        self.min_points = min_points
        self.margin = margin
        self.max_retries = max_retries

    def _get_reduction_schedule(self, init_points: float) -> list:
        """
        计算几何级数下降的调度表
        返回一个列表，包含每一层应该保留的点数
        """
        # 1. 确保输入是浮点数，否则 torch.log 会报错
        # 2. 在对数空间线性插值
        log_steps = torch.linspace(
            torch.log(torch.tensor(float(init_points))),
            torch.log(torch.tensor(float(self.target_points))),
            steps=self.convlayer_nums + 1,
        )

        # 3. 转换回线性空间
        steps = torch.exp(log_steps)

        # 4. 取整并转换为长整型 (PyTorch 使用 .int() 或 .long())
        steps = torch.round(steps).long()

        # 5. 确保数值合法性
        # 使用 torch.clamp 确保最小值为 1
        steps = torch.clamp(steps, min=1)

        # 6. 返回最初的点数到第L层的目标点数 (去掉 steps[0])
        return steps[0:].tolist()

    def _monte_carlo_fill_tensor(self, positions, belonging, target_n):
        """
        使用带距离拒绝的蒙特卡洛采样填充点云 (Tensor版本)。

        Args:
            positions (torch.Tensor): (N, 3) 粒子的坐标
            belonging (torch.Tensor): (N, 1) 或 (N,) 粒子所属的点云ID
            target_n (int): 目标点数
            margin (float): 边界框外扩的边距
            max_retries (int): 单次撒点的最大重试次数

        Returns:
            out_positions (torch.Tensor): (M * target_n, 3) 填充后的坐标
            out_belonging (torch.Tensor): (M * target_n, 1) 对应的点云ID
            global_indices (torch.Tensor): (M * target_n,) 每个点在原始positions中的全局索引，新增点为-1
        """
        device = positions.device

        unique_groups = torch.unique(belonging)

        # 预分配结果列表
        result_pos_list = []
        result_idx_list = []
        result_global_indices_list = []  # 存储全局索引

        # 使得目标点数不小于设定值
        target_n = max(target_n, self.min_points)

        # 创建全局索引映射
        # 对于每个点，我们需要知道它在原始positions中的位置
        # 我们假设belonging和positions的顺序是对应的
        num_points = positions.shape[0]

        # 计算每个点云中的点数和偏移量
        group_counts = []
        group_offsets = []
        offset = 0
        for group_id in unique_groups:
            mask = belonging == group_id
            count = mask.sum().item()
            group_counts.append(count)
            group_offsets.append(offset)
            offset += count

        # 对每个点云进行处理
        for group_idx, group_id in enumerate(unique_groups):
            # 1. 获取当前组的坐标和全局索引
            mask = belonging == group_id
            coords = positions[mask]  # (Current_N, 3)
            current_n = coords.shape[0]

            # 获取这个点云中所有点的全局索引
            # 我们通过计算在原始positions中的位置来获取
            start_idx = group_offsets[group_idx]
            group_global_indices = torch.arange(
                start_idx, start_idx + current_n, device=device
            )

            # Case A: 点数足够，进行下采样
            if current_n >= target_n:
                # 随机选择 target_n 个索引，无放回
                perm = torch.randperm(current_n, device=device)[:target_n]
                selected_coords = coords[perm]
                selected_global_indices = group_global_indices[perm]

                result_pos_list.append(selected_coords)
                result_idx_list.append(
                    torch.full((target_n, 1), group_id, device=device)
                )
                result_global_indices_list.append(selected_global_indices)
                continue

            # Case B: 点数不足，需要蒙特卡洛填充
            # 2. 确定撒点范围 (Bounding Box)
            min_bound = torch.min(coords, dim=0)[0] - self.margin
            max_bound = torch.max(coords, dim=0)[0] + self.margin

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
            points_pool = coords.clone()
            points_needed = target_n - current_n

            # 临时存储生成的点
            new_points = []

            for _ in range(points_needed):
                success = False
                current_threshold = min_dist_threshold.clone()

                # 尝试 max_retries 次
                for retry in range(self.max_retries):
                    # 在 Box 内随机生成一个点 (1, 3)
                    candidate = (
                        torch.rand(1, 3, device=device) * (max_bound - min_bound)
                        + min_bound
                    )

                    # 计算到池子中所有点的距离
                    dists = torch.cdist(candidate, points_pool)

                    # 检查是否太近
                    if torch.all(dists > current_threshold):
                        points_pool = torch.cat([points_pool, candidate], dim=0)
                        new_points.append(candidate)
                        success = True
                        break

                # 如果重试多次仍失败，说明太挤了，降低阈值并强制插入
                if not success:
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

            # 对于新增加的点，索引为-1（表示不是原始点）
            # 原始点的全局索引保持不变
            new_indices = torch.full(
                (points_needed,), -1, device=device, dtype=torch.long
            )
            all_global_indices = torch.cat([group_global_indices, new_indices])

            result_pos_list.append(full_coords)
            result_idx_list.append(torch.full((target_n, 1), group_id, device=device))
            result_global_indices_list.append(all_global_indices)

        # 最终合并
        out_positions = torch.cat(result_pos_list, dim=0)
        out_belonging = torch.cat(result_idx_list, dim=0)
        out_global_indices = torch.cat(result_global_indices_list, dim=0)

        return out_positions, out_belonging, out_global_indices

    def __call__(
        self, positions: torch.Tensor, channel: torch.Tensor, belonging: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        # 使用 torch.unique 获取每个元素及其出现次数
        unique_groups, counts = torch.unique(belonging, return_counts=True)

        # 找到最高重复次数
        max_count = torch.max(counts).item()

        # 使用填充算法将每一片点云中的点数都填充到与具有最多点数的点云空间相同
        filled_positions, filled_belonging, global_indices = (
            self._monte_carlo_fill_tensor(positions, belonging, max_count)
        )

        # 依据点云中点的数目生成点数衰减列表
        reduction_schedule = self._get_reduction_schedule(max_count)

        # 扩展通道矩阵 channel 以匹配 filled_positions 的形状
        # 获取原始通道矩阵的列数
        num_channels = channel.size(1)

        # 创建扩展后的通道矩阵，初始化为0
        expanded_channel = torch.zeros(
            filled_positions.size(0),
            num_channels,
            device=channel.device,
            dtype=channel.dtype,
        )

        # 使用全局索引将原始通道信息复制到新矩阵中
        # 只有全局索引 >= 0 的点才是原始点
        valid_mask = global_indices >= 0
        valid_indices = global_indices[valid_mask].long()

        # 将原始通道信息复制到对应位置
        expanded_channel[valid_mask] = channel[valid_indices]

        # 对于新增加的点（索引为-1），通道值已经初始化为0，不需要额外操作

        return filled_positions, expanded_channel, reduction_schedule
