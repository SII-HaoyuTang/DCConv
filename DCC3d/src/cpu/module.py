import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.dcconv3d_batchnrom import Dcconv3dBatchnorm
from DCC3d.src.cpu.multi_layer_resblock import ResidualDCConv3dNetwork
from DCC3d.src.cpu.selector import SelectorConfig, SelectorType, get_reduction_schedule

from .data.dataset import PointCloudCollater, PointCloudQM9Dataset, PointCloudTransform


def get_dataloader():
    # 数据集路径
    points_csv = "./data/qm9.csv"
    indices_csv = "./data/qm9_indices.csv"

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


class DCConvNet(nn.Module):
    """
    Multi-layer DCConv network with point reduction schedule.

    This network progressively reduces the number of points from N to 1 through multiple layers,
    each consisting of {DCConv, BatchNorm, Residual connections}.
    """

    def __init__(
        self,
        in_channels: int = 4,  # Default for QM9 dataset features
        hidden_channels: list[int] = [64, 128, 256, 512],
        out_channels: int = 1,  # For regression tasks
        num_layers: int = 5,
        selector_config: SelectorConfig | None = None,
        N: int = 5,  # Polynomial basis functions
        L: int = 3,  # Angular basis functions
        M: int = 3,  # Radial basis functions
        use_PCA: bool = True,
    ):
        """
        Initialize the multi-layer DCConv network.

        Args:
            in_channels: Input feature channels
            hidden_channels: List of hidden layer channels
            out_channels: Output channels (typically 1 for regression)
            num_layers: Number of layers in the network
            selector_config: Configuration for point selection strategy
            N, L, M: Kernel polynomial parameters
            use_PCA: Whether to use PCA in coordinate transformation
        """
        super(DCConvNet, self).__init__()

        # Default selector configuration if not provided
        if selector_config is None:
            selector_config = SelectorConfig(type=SelectorType.KNN, n=16)

        self.selector_config = selector_config
        self.num_layers = num_layers
        self.N, self.L, self.M = N, L, M
        self.use_PCA = use_PCA

        # Build channel progression: in -> hidden -> out
        all_channels = [in_channels] + hidden_channels + [out_channels]

        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(len(all_channels) - 1):
            # DCConv layer
            layer = DistanceContainedConv3d(
                in_channels=all_channels[i],
                out_channels=all_channels[i + 1],
                config=selector_config,
                N=N,
                L=L,
                M=M,
                use_PCA=use_PCA,
            )
            self.layers.append(layer)

            # BatchNorm layer (except for the last layer)
            if i < len(all_channels) - 2:
                batch_norm = Dcconv3dBatchnorm(num_features=all_channels[i + 1])
                self.batch_norms.append(batch_norm)

        # Activation function
        self.activation = nn.ReLU(inplace=True)

        # Final regression head (to reduce from final hidden size to output)
        self.final_head = nn.Linear(all_channels[-1], out_channels)

    def forward(
        self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with progressive point reduction.

        Args:
            position_matrix: Point coordinates (N, 3)
            channel_matrix: Point features (N, in_channels)

        Returns:
            output: Final prediction (1, out_channels) for regression
        """
        N = position_matrix.shape[0]

        # Generate reduction schedule from N points to 1 point
        reduction_schedule = get_reduction_schedule(
            n_input=N, n_target=1, num_layers=self.num_layers
        )

        # Start with all points
        current_pos = position_matrix
        current_features = channel_matrix

        # Progressive reduction through layers
        for i, layer in enumerate(self.layers):
            # Determine how many points to select for this layer
            if i < len(reduction_schedule):
                n_select = reduction_schedule[i]
            else:
                # Last layer or beyond schedule, select 1 point
                n_select = 1

            # Apply DCConv layer with point reduction
            centers, features = layer(current_pos, current_features, n_select)

            # Apply batch normalization (except for last layer)
            if i < len(self.batch_norms):
                # BatchNorm1d expects (N, C) where N is batch size
                # Our features are (n_select, channels)
                features = self.batch_norms[i](features)

                # Apply activation
                features = self.activation(features)

            # Update current state for next layer
            current_pos = centers  # Use selected centers as new positions
            current_features = features

        # At this point, we should have (1, final_channels)
        # Apply final regression head
        if current_features.shape[0] == 1:
            output = self.final_head(current_features)  # (1, out_channels)
        else:
            # If somehow we have more than 1 point, take mean
            pooled_features = current_features.mean(
                dim=0, keepdim=True
            )  # (1, channels)
            output = self.final_head(pooled_features)  # (1, out_channels)

        return output.squeeze(0)  # Return (out_channels,) for single sample


class DCConvResNet(nn.Module):
    """
    Residual DCConv network with point reduction schedule.

    This network uses the existing ResidualDCConv3dNetwork with progressive point reduction,
    providing more sophisticated residual connections across multiple layers.
    """

    def __init__(
        self,
        in_channels: int = 4,
        block_configs: list[list[int]] = [
            [4, 64, 64],
            [64, 128, 128],
            [128, 256, 256],
            [256, 512],
        ],
        out_channels: int = 1,
        num_layers: int = 4,
        selector_config: SelectorConfig | None = None,
        N: int = 5,
        L: int = 3,
        M: int = 3,
        use_PCA: bool = True,
    ):
        """
        Initialize the residual DCConv network.

        Args:
            in_channels: Input feature channels
            block_configs: List of channel configurations for each residual block
            out_channels: Output channels for final prediction
            num_layers: Number of layers for reduction schedule
            selector_config: Point selection configuration
            N, L, M: Kernel polynomial parameters
            use_PCA: Whether to use PCA in coordinate transformation
        """
        super(DCConvResNet, self).__init__()

        if selector_config is None:
            selector_config = SelectorConfig(type=SelectorType.KNN, n=16)

        self.num_layers = num_layers

        # Create the residual network
        self.resnet = ResidualDCConv3dNetwork(
            block_configs=block_configs,
            config=selector_config,
            N=N,
            L=L,
            M=M,
            use_PCA=use_PCA,
        )

        # Final head for regression
        final_channels = block_configs[-1][-1]  # Last channel in last block
        self.final_head = nn.Linear(final_channels, out_channels)

    def forward(
        self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with residual connections and point reduction.

        Args:
            position_matrix: Point coordinates (N, 3)
            channel_matrix: Point features (N, in_channels)

        Returns:
            output: Final prediction (out_channels,)
        """
        N = position_matrix.shape[0]

        # Generate reduction schedule
        reduction_schedule = get_reduction_schedule(
            n_input=N, n_target=1, num_layers=self.num_layers
        )

        # For simplicity, we'll use the last value in the schedule for all layers
        # In practice, you might want to distribute the reduction across blocks
        final_n_select = reduction_schedule[-1] if reduction_schedule else 1

        # Apply residual network
        features = self.resnet(position_matrix, channel_matrix, final_n_select)

        # Global pooling if we still have multiple points
        if features.shape[0] > 1:
            features = features.mean(dim=0, keepdim=True)  # (1, channels)

        # Final prediction
        output = self.final_head(features)  # (1, out_channels)
        return output.squeeze(0)


def create_model(model_type: str = "dcconv", **kwargs) -> nn.Module:
    """
    Factory function to create different types of DCConv models.

    Args:
        model_type: Type of model to create ("dcconv" or "resnet")
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Initialized model
    """
    if model_type.lower() == "dcconv":
        return DCConvNet(**kwargs)
    elif model_type.lower() == "resnet":
        return DCConvResNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_and_data():
    """
    Convenience function to get both the model and dataloaders.

    Returns:
        tuple: (model, train_loader, val_loader, test_loader)
    """
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloader()

    # Create model with appropriate input channels for QM9 dataset
    model = create_model(
        model_type="dcconv",
        in_channels=4,  # QM9 features: mass, valence, radius, charge
        hidden_channels=[64, 128, 256, 512],
        out_channels=1,  # Regression task
        num_layers=5,
    )

    return model, train_loader, val_loader, test_loader
