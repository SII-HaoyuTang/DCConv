import torch
import torch.nn as nn

from .aggregation import AggregationLayer
from .kernel import DCConv3dKernelPolynomials
from .selector import SelectorConfig, SelectorFactory
from .transformation import CoordinateTransformer


class DistanceContainedConv3d(nn.Module):
    """
    A 3D convolutional layer that incorporates distance information for feature transformation and aggregation.
    The layer processes input tensors through a series of steps including neighbor selection, coordinate transformation,
    kernel weight generation, and feature aggregation. The output is a tensor that combines the transformed features
    with the centers of the original positions.

    Detailed description:
    This class implements a 3D convolutional layer designed to handle spatial data with an emphasis on maintaining
    distance information between points. It utilizes a coordinate transformer, a selector for neighbor points,
    a kernel for generating dynamic weights based on spherical coordinates, and an aggregation layer to combine
    features. The primary goal is to enhance the representation of spatial relationships in the input data,
    making it suitable for tasks that require understanding of geometric structures.
    """
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
        self.kernel = DCConv3dKernelPolynomials(out_channels, in_channels, N, L, M)
        self.aggregation = AggregationLayer()

    def forward(
        self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Summary:
        This function performs a forward pass through the network, executing several steps including
        neighbor selection, coordinate transformation, kernel weight generation, and feature aggregation.
        The input consists of a position matrix and a channel matrix, and the output is a tuple containing
        the centers and the final output tensor after processing.

        Parameters:
        - position_matrix (torch.Tensor): The input position matrix for the points.
        - channel_matrix (torch.Tensor): The input channel matrix for the features.

        Returns:
        - tuple[torch.Tensor, torch.Tensor]: A tuple containing the centers and the processed output tensor.
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
        # print(spherical_coords[:, :, 0])
        kernel_weights = self.kernel.forward(spherical_coords)  # Shape: (Co, Ci, N, k)
        print(f"kernel_weights.shape: {kernel_weights.shape}")

        # Step 4: 特征聚合 (Feature Aggregation)
        # 需要将 local_features 从 (N, k, Ci) 转换为 (Ci, N, k)
        local_features_transposed = local_features.permute(2, 0, 1)  # (Ci, N, k)

        # 使用聚合层进行加权求和
        output = self.aggregation.forward(
            local_features_transposed, kernel_weights
        )  # Shape: (Co, N)
        output = output.permute(1, 0)  # (N, Co)

        return centers, output,
