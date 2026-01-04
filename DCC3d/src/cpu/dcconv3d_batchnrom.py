import torch
import torch.nn as nn


class Dcconv3dBatchNorm(nn.Module):
    """
    A module that applies 1D batch normalization to a 3D convolutional layer's output.

    This class is designed to be used as part of a neural network model, specifically
    for normalizing the outputs of 3D convolutional layers. It leverages the `nn.BatchNorm1d`
    from PyTorch to perform batch normalization, which helps in stabilizing and accelerating
    the training process by normalizing the input to each layer.

    :param num_features: The number of features or channels in the input tensor.
    :type num_features: int
    :returns: Normalized tensor after applying 1D batch normalization.
    :rtype: torch.Tensor
    """

    def __init__(self, num_features):
        super(Dcconv3dBatchNorm, self).__init__()
        self.batchnorm1d = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor, num_points: int) -> torch.Tensor:
        N, M = x.shape
        x = x.reshape(N//num_points, num_points, M).permute(0, 2, 1)

        out = self.batchnorm1d(x)

        out = out.permute(0, 2, 1)

        # 步骤2: 重塑为 (N, M)
        out = out.reshape(N, M)

        return out


class Dcconv3dLayerNorm(nn.Module):
    def __init__(self, norm_shape):
        super(Dcconv3dLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=norm_shape)

    def forward(self, x: torch.Tensor, num_points: int) -> torch.Tensor:
        N, M = x.shape
        x = x.reshape(N//num_points, num_points, M).permute(0, 2, 1)

        out = self.layernorm(x)

        out = out.permute(0, 2, 1)

        # 步骤2: 重塑为 (N, M)
        out = out.reshape(N, M)

        return out
