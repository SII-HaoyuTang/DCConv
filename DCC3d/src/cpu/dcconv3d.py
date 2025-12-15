import torch
import torch.nn as nn
from selector import SelectorConfig, SelectorFactory
from kernel.DCConv3d_kernel import DCConv3dKernelPolynomials
from transformation.coordinate_transformer import CoordinateTransformer
from aggregation

class DistanceContainedConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: SelectorConfig, N: int , L: int, M: int, use_PCA: bool = True):
        super(DistanceContainedConv3d, self).__init__()
        self.cotrans = CoordinateTransformer(use_pca=use_PCA)
        self.selector = SelectorFactory.get_selector(config)
        self.kernel = DCConv3dKernelPolynomials(in_channels, out_channels, N, L, M)
        self.

    def forward(self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor):
