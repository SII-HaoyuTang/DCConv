import torch
import torch.nn as nn


class Dcconv3dBatchnorm(nn.Module):
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
        super(Dcconv3dBatchnorm, self).__init__()
        self.batchnorm1d = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.batchnorm1d(x)

        return out