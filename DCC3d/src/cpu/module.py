import torch
import torch.nn as nn

from DCC3d.src.cpu.selector.pre_selector import PreSelector
from DCC3d.src.cpu.selector import SelectorConfig, SelectorType
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.dcconv3d_batchnrom import Dcconv3dBatchnorm

class ResnetBlock(nn.Module):
    def __init__(self, init_channels: int, out_channels: int, N: int, K: int, M: int, conv_nums: int, selector_config: SelectorConfig | None = None):
        super(ResnetBlock,self).__init__()

        if not selector_config:
            selector_config = SelectorConfig(type=SelectorType.KNN)

        self.dcconv3d_1 = DistanceContainedConv3d(init_channels, out_channels, selector_config, N, K, M, conv_nums)
        self.batchnorm_1 = Dcconv3dBatchnorm(out_channels)

        self.dcconv3d_2 = DistanceContainedConv3d(out_channels, out_channels, selector_config, N, K, M, conv_nums)
        self.batchnorm_2 = Dcconv3dBatchnorm(out_channels)

        self.batchnorm_resnet = Dcconv3dBatchnorm(out_channels)

    def forward(self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor, n_select: list[int]):

        out_position, out_channel, resnet_channel = self.dcconv3d_1(position_matrix, channel_matrix, n_select[0], n_select[1])
        out_channel = self.batchnorm_1(out_channel)

        out_position, out_channel, resnet_channel = self.dcconv3d_2(out_position, out_channel, n_select[1], n_select[2], resnet_channel)
        out_channel = self.batchnorm_2(out_channel)

        out_channel += self.batchnorm_resnet(resnet_channel)

        return out_position, out_channel


class LinearLayer(nn.Module):
    def __init__(self, input_features: int, out_features: int):
        super(LinearLayer, self).__init__()

        self.layer_1 = nn.Linear(input_features, input_features//4)
        self.activation_1 = nn.ReLU()

        self.layer_2 = nn.Linear(input_features//4, input_features//8)
        self.activation_2 = nn.ReLU()

        self.layer_3 = nn.Linear(input_features//8, out_features)

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        out = self.activation_1(out)

        out = self.layer_2(out)
        out = self.activation_2(out)

        out = self.layer_3(out)

        return out



class DCConvNet(nn.Module):
    def __init__(self, num_features: int):
        super(DCConvNet, self).__init__()

        self.pre_selector = PreSelector(1, 7)

        selector_config = SelectorConfig(type=SelectorType.KNN)

        self.init_conv = DistanceContainedConv3d(num_features, 16, selector_config, 3, 2, 2, 9, use_resnet=False)
        self.init_batchnorm = Dcconv3dBatchnorm(16)

        self.resnetblock_1 = ResnetBlock(16, 32, 4, 3, 3, 9, selector_config)

        self.resnetblock_2 = ResnetBlock(32, 64, 4, 3, 3, 6, selector_config)

        self.resnetblock_3 = ResnetBlock(64, 128, 5, 4, 4, 3, selector_config)

        self.linear = LinearLayer(128, 1)

    def forward(
        self, position_matrix: torch.Tensor, channel_matrix: torch.Tensor, belonging: torch.Tensor
    ) -> torch.Tensor:
        # Generate reduction schedule from N points to 1 point
        position_matrix, channel_matrix, reduction_schedule = self.pre_selector(position_matrix, channel_matrix, belonging)

        out_pos, out_ch = self.init_conv(position_matrix, channel_matrix, reduction_schedule[0], reduction_schedule[1])
        out_ch = self.init_batchnorm(out_ch)

        out_pos, out_ch = self.resnetblock_1(out_pos, out_ch, [reduction_schedule[1], reduction_schedule[2], reduction_schedule[3]])

        out_pos, out_ch = self.resnetblock_2(out_pos, out_ch, [reduction_schedule[3], reduction_schedule[4], reduction_schedule[5]])

        out_pos, out_ch = self.resnetblock_3(out_pos, out_ch, [reduction_schedule[5], reduction_schedule[6], reduction_schedule[7]])

        out = self.linear(out_ch)

        return out