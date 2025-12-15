import torch
import torch.nn as nn

from .aggregation import AggregationLayer
from .kernel import DCConv3dKernelPolynomials
from .selector import SelectorConfig, SelectorFactory
from .transformation import CoordinateTransformer


class DistanceContainedConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SelectorConfig,
        N: int,
        L: int,
        M: int,
        use_PCA: bool = True,
    ):
        super(DistanceContainedConv3d, self).__init__()
        self.cotrans = CoordinateTransformer(use_pca=use_PCA)
        self.selector = SelectorFactory.get_selector(config)
        self.kernel = DCConv3dKernelPolynomials(in_channels, out_channels, N, L, M)
        self.aggregation = AggregationLayer()

    def forward(
        self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播 (Forward Pass)

        Args:
            position_matrix (torch.Tensor): 点云坐标矩阵
                Shape: (N, 3), 其中 N 是点的数量，3 代表 (x, y, z)
            channel_matrix (torch.Tensor): 点的属性特征矩阵
                Shape: (N, Ci), 其中 Ci 是输入通道数

        Returns:
            output (torch.Tensor): 卷积后的特征
                Shape: (Co, N), 其中 Co 是输出通道数
        """
        # Step 1: 邻居选择 (Neighbor Selection)
        # 使用选择器为每个点选择邻居
        neighbor_indices = self.selector(position_matrix)  # Shape: (N, k)

        # Step 2: 坐标转换 (Coordinate Transformation)
        # 将绝对坐标转换为相对的球极坐标
        spherical_coords, centers, eigenvalues, local_features = self.cotrans.forward(
            global_coords=position_matrix,
            neighbor_indices=neighbor_indices,
            global_features=channel_matrix,
        )
        # spherical_coords: (N, k, 3) - 球极坐标 (r, θ, φ)
        # local_features: (N, k, Ci) - 局部属性特征

        # Step 3: 核权重生成 (Kernel Weight Generation)
        # 基于球极坐标生成动态卷积核权重
        kernel_weights = self.kernel.forward(spherical_coords)  # Shape: (Co, Ci, N, k)

        # Step 4: 特征聚合 (Feature Aggregation)
        # 需要将 local_features 从 (N, k, Ci) 转换为 (Ci, N, k)
        local_features_transposed = local_features.permute(2, 0, 1)  # (Ci, N, k)

        # 使用聚合层进行加权求和
        output = self.aggregation.forward(
            local_features_transposed, kernel_weights
        )  # Shape: (Co, N)
        output = output.permute(1, 0)  # (N, Co)

        return output
