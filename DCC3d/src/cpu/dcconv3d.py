import torch
import torch.nn as nn
from .aggregation import AggregationLayer
from .kernel import DCConv3dKernelPolynomials
from .selector import SelectorConfig, SelectorFactory
from .transformation import CoordinateTransformer


class DistanceContainedConv3d(nn.Module):
    """
    A 3D convolutional layer that incorporates distance information for more effective feature extraction in point
    cloud data.

    This class defines a 3D convolution operation that takes into account the spatial relationships and distances
    between points. It is designed to work with point cloud data, where each point has a position and associated
    features (channels). The layer includes steps for neighbor selection, coordinate transformation, kernel weight
    generation, and feature aggregation. Optionally, it can also incorporate a ResNet-like residual connection for
    improved gradient flow during training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SelectorConfig,
        N: int,
        L: int,
        M: int,
        conv_num: int,
        use_PCA: bool = True,
        use_resnet: bool = True,
    ):
        """
        Initializes a new instance of the DistanceContainedConv3d class, which is designed to perform 3D convolutions
        with distance-contained kernels. This class integrates a coordinate transformer, a selector, and an aggregation
        layer, and optionally uses PCA and ResNet.

        Parameters:
        ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        config: SelectorConfig
            Configuration for the selector.
        N: int
            A parameter for the DCConv3dKernelPolynomials.
        L: int
            A parameter for the DCConv3dKernelPolynomials.
        M: int
            A parameter for the DCConv3dKernelPolynomials.
        conv_num: int
            The points number in one local convolution.
        use_PCA: bool, optional
            Whether to use PCA in the coordinate transformation. Default is True.
        use_resnet: bool, optional
            Whether to include a ResNet block. Default is True.

        Attributes:
        ----------
        conv_nums: int
            The points number in one local convolution.
        cotrans: CoordinateTransformer
            Transformer for coordinates, possibly using PCA.
        selector: Selector
            Selector object created from the given configuration.
        kernel: DCConv3dKernelPolynomials
            Kernel for the 3D convolution operation.
        aggregation: AggregationLayer
            Layer for aggregating the results of the convolution.
        use_resnet: bool
            Indicates if a ResNet block is used.
        in_channels: int
            The number of input channels, set when use_resnet is True.
        out_channels: int
            The number of output channels, set when use_resnet is True.
        """
        super(DistanceContainedConv3d, self).__init__()
        self.conv_nums = conv_num
        self.cotrans = CoordinateTransformer(use_pca=use_PCA)
        self.selector = SelectorFactory.get_selector(config)
        self.kernel = DCConv3dKernelPolynomials(out_channels, in_channels, N, L, M)
        self.aggregation = AggregationLayer()
        self.use_resnet = use_resnet
        self.bias = torch.nn.Parameter(torch.zeros(out_channels,))
        if self.use_resnet:
            self.in_channels = in_channels
            self.out_channels = out_channels

    def forward(
        self,
        position_matrix: torch.Tensor,
        channel_matrix: torch.Tensor,
        space_points_num: int,
        outpoint_num: int,
        resnet_channel: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """ """
        if space_points_num < outpoint_num:
            raise ValueError(
                f"space_points_num must be greater than outpoint_num, but got {space_points_num} and {outpoint_num}"
            )

        # Step 1: 邻居选择 (Neighbor Selection)
        conv_num = space_points_num - outpoint_num + 1
        conv_num = min(conv_num, self.conv_nums)

        # 按照空间数目生成线性间隔的张量,之后重复每个值空间中点数次，最后生成归属向量belonging
        space_num = int(position_matrix.shape[0] / space_points_num)
        linspace_tensor = torch.linspace(
            1, space_num, space_num, device=position_matrix.device
        )
        belonging = linspace_tensor.repeat_interleave(space_points_num)

        neighbor_indices = self.selector(
            position_matrix, belonging, conv_num, outpoint_num
        )  # Shape: (outpoint_num, k)

        # Step 2: 坐标转换 (Coordinate Transformation)
        # 将绝对坐标转换为相对的球极坐标
        # Note: We need to use the selected subset for coordinate transformation
        spherical_coords, centers, _, local_features = self.cotrans.forward(
            global_coords=position_matrix,
            neighbor_indices=neighbor_indices,
            global_features=channel_matrix,
        )
        # spherical_coords: (outpoint_num, k, 3) - 球极坐标 (r, θ, φ)
        # local_features: (outpoint_num, k, Ci) - 局部属性特征
        # centers: (outpoint_num, 3) - 选择的中心点坐标

        # Step 3: 核权重生成 (Kernel Weight Generation)
        # 基于球极坐标生成动态卷积核权重
        kernel_weights = self.kernel.forward(
            spherical_coords
        )  # Shape: (Co, Ci, outpoint_num, k)

        # Step 4: 特征聚合 (Feature Aggregation)
        # 需要将 local_features 从 (outpoint_num, k, Ci) 转换为 (Ci, outpoint_num, k)
        local_features = local_features.permute(2, 0, 1)  # (Ci, outpoint_num, k)

        # 使用聚合层进行加权求和
        output = self.aggregation.forward(
            local_features, kernel_weights
        )  # Shape: (Co, outpoint_num)
        output = output.permute(1, 0) + self.bias # (outpoint_num, Co)

        if self.use_resnet:
            # if resnet_channel is given by last DCConv, resnet continue. Else the shortcut has been done, start a
            # new resnet.
            if isinstance(resnet_channel, torch.Tensor):
                resnet_channel = self.cotrans.extract_local_features(
                    resnet_channel, neighbor_indices
                )
                resnet_channel = resnet_channel.permute(2, 0, 1)
            else:
                resnet_channel = local_features.clone()

            # Step 3': 残差核权重生成 (Resnet Kernel Weight Generation)
            resnet_kernel_weights = torch.ones(
                self.out_channels,
                self.in_channels,
                outpoint_num * space_num,
                conv_num,
                device=resnet_channel.device,
            )  # Shape: (Co, Ci, outpoint_num, k)

            # Step 4‘: 特征聚合 (Feature Aggregation)
            # 使用聚合层进行加权求和
            resnet_output = self.aggregation.forward(
                resnet_channel, resnet_kernel_weights
            )  # Shape: (Co, outpoint_num)
            resnet_output = resnet_output.permute(1, 0)  # (outpoint_num, Co)
            return (
                centers,
                output,
                resnet_output,
            )
        else:
            return (
                centers,
                output,
            )
