import torch
import torch.nn as nn

from DCC3d.src.cpu.selector.pre_selector import PreSelector
from DCC3d.src.cpu.selector import SelectorConfig, SelectorType
from DCC3d.src.cpu.dcconv3d import DistanceContainedConv3d
from DCC3d.src.cpu.dcconv3d_batchnrom import Dcconv3dLayerNorm


# ==================== 参数初始化方法 ====================
def init_weights_kaiming_normal(module: nn.Module):
    """使用Kaiming正态分布初始化权重"""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, DistanceContainedConv3d)):
        # 使用较小的负斜率，避免梯度爆炸
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu", a=0.1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)  # 避免零初始化
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0.01)  # 避免零初始化
    elif isinstance(module, Dcconv3dLayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)  # 初始化为1
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)  # 避免零初始化
def init_weights_xavier_uniform(module: nn.Module):
    """使用Xavier均匀分布初始化权重"""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight, gain=0.5)  # 降低增益
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)  # 避免零初始化
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0.01)  # 避免零初始化
    elif isinstance(module, Dcconv3dLayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)  # 初始化为1
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)  # 避免零初始化
def apply_weight_decay(
    module: nn.Module, weight_decay: float = 1e-4, skip_list: tuple = ()
):
    """为模块参数应用权重衰减（L2正则化）"""
    decay_params = []
    no_decay_params = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name in skip_list
            or any(skip_name in name for skip_name in skip_list)
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def apply_l1_regularization(module: nn.Module, l1_lambda: float = 1e-7):
    """为模块参数应用L1正则化"""
    l1_reg = torch.tensor(0.0, requires_grad=True)

    for param in module.parameters():
        if param.requires_grad:
            l1_reg = l1_reg + torch.norm(param, p=1)

    return l1_lambda * l1_reg


class ResnetBlock(nn.Module):
    def __init__(
        self,
        init_channels: int,
        out_channels: int,
        N: int,
        K: int,
        M: int,
        conv_nums: int,
        selector_config: SelectorConfig | None = None,
    ):
        super(ResnetBlock, self).__init__()

        if not selector_config:
            selector_config = SelectorConfig(type=SelectorType.KNN)

        self.dcconv3d_1 = DistanceContainedConv3d(
            init_channels, out_channels, selector_config, N, K, M, conv_nums
        )
        self.layernorm_1 = Dcconv3dLayerNorm(out_channels)
        self.silu1 = nn.SiLU()
        self.dropout1 = nn.Dropout(0.1)

        self.dcconv3d_2 = DistanceContainedConv3d(
            out_channels, out_channels, selector_config, N, K, M, conv_nums
        )
        self.layernorm_2 = Dcconv3dLayerNorm(out_channels)
        self.silu2 = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)

        # 初始化权重
        self.init_weights()

    def init_weights(self, init_method: str = "kaiming"):
        """初始化模块权重"""
        if init_method == "kaiming":
            init_weights_kaiming_normal(self)
        elif init_method == "xavier":
            init_weights_xavier_uniform(self)
        else:
            raise ValueError(f"不支持的初始化方法: {init_method}")

    def forward(
        self,
        position_matrix: torch.Tensor,
        channel_matrix: torch.Tensor,
        n_select: list[int],
    ):
        out_position, out_channel, resnet_channel = self.dcconv3d_1(
            position_matrix, channel_matrix, n_select[0], n_select[1]
        )
        out_channel = self.layernorm_1(out_channel, n_select[1])
        out_channel = self.silu1(out_channel)
        out_channel = self.dropout1(out_channel)

        out_position, out_channel, resnet_channel = self.dcconv3d_2(
            out_position, out_channel, n_select[1], n_select[2], resnet_channel
        )
        out_channel = self.layernorm_2(out_channel, n_select[2])
        out_channel = self.silu2(out_channel)
        out_channel = self.dropout2(out_channel)

        out_channel += resnet_channel

        return out_position, out_channel


class LinearLayer(nn.Module):
    def __init__(self, input_features: int, out_features: int):
        super(LinearLayer, self).__init__()

        self.layer_1 = nn.Linear(input_features, input_features // 4)
        self.activation_1 = nn.SiLU()
        self.dropout1 = nn.Dropout(0.5)

        self.layer_2 = nn.Linear(input_features // 4, input_features // 8)
        self.activation_2 = nn.SiLU()
        self.dropout2 = nn.Dropout(0.5)

        self.layer_3 = nn.Linear(input_features // 8, out_features)

        # 初始化权重
        self.init_weights()

    def init_weights(self, init_method: str = "kaiming"):
        """初始化模块权重"""
        if init_method == "kaiming":
            init_weights_kaiming_normal(self)
        elif init_method == "xavier":
            init_weights_xavier_uniform(self)
        else:
            raise ValueError(f"不支持的初始化方法: {init_method}")

    def forward(self, x: torch.Tensor):
        out = self.layer_1(x)
        out = self.activation_1(out)
        out = self.dropout1(out)

        out = self.layer_2(out)
        out = self.activation_2(out)
        out = self.dropout2(out)

        out = self.layer_3(out)

        return out


class DCConvNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        init_method: str = "kaiming",
        use_weight_decay: bool = True,
        weight_decay: float = 1e-4,
    ):
        super(DCConvNet, self).__init__()

        self.pre_selector = PreSelector(1, 3)

        selector_config = SelectorConfig(type=SelectorType.KNN)

        self.init_conv = DistanceContainedConv3d(
            num_features, 32, selector_config, 4, 3, 3, 10
        )
        self.init_layernorm = Dcconv3dLayerNorm(32)

        self.resnetblock_1 = ResnetBlock(32, 128, 3, 2, 2, 5, selector_config)
        #
        # self.resnetblock_2 = ResnetBlock(64, 64, 2, 1, 1, 6, selector_config)
        #
        # self.resnetblock_3 = ResnetBlock(64, 128, 2, 1, 1, 3, selector_config)

        self.linear = LinearLayer(128, 1)

        # 初始化权重
        self.init_weights(init_method)

        # 正则化设置
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay
        self.l1_lambda = 1e-5

    def init_weights(self, init_method: str = "kaiming"):
        """初始化整个模型的权重"""
        if init_method == "kaiming":
            init_weights_kaiming_normal(self)
        elif init_method == "xavier":
            init_weights_xavier_uniform(self)
        else:
            raise ValueError(f"不支持的初始化方法: {init_method}")

    def get_regularization_params(
        self, skip_list: tuple = ("bias", "LayerNorm.weight")
    ):
        """获取带有正则化的优化器参数组"""
        if self.use_weight_decay:
            return apply_weight_decay(self, self.weight_decay, skip_list)
        else:
            return [{"params": self.parameters(), "weight_decay": 0.0}]

    def get_l1_regularization_loss(self):
        """计算L1正则化损失"""
        return apply_l1_regularization(self, self.l1_lambda)

    def forward(
        self,
        position_matrix: torch.Tensor,
        channel_matrix: torch.Tensor,
        belonging: torch.Tensor,
    ) -> torch.Tensor:
        # Generate reduction schedule from N points to 1 point
        position_matrix, channel_matrix, reduction_schedule = self.pre_selector(
            position_matrix, channel_matrix, belonging
        )

        out_pos, out_ch, res_ch = self.init_conv(
            position_matrix,
            channel_matrix,
            reduction_schedule[0],
            reduction_schedule[1],
        )
        out_ch = self.init_layernorm(out_ch, reduction_schedule[1])
        out_ch = out_ch + res_ch

        out_pos, out_ch = self.resnetblock_1(
            out_pos,
            out_ch,
            [reduction_schedule[1], reduction_schedule[2], reduction_schedule[3]],
        )

        # out_pos, out_ch = self.resnetblock_2(
        #     out_pos,
        #     out_ch,
        #     [reduction_schedule[3], reduction_schedule[4], reduction_schedule[5]],
        # )
        #
        # out_pos, out_ch = self.resnetblock_3(
        #     out_pos,
        #     out_ch,
        #     [reduction_schedule[5], reduction_schedule[6], reduction_schedule[7]],
        # )

        out = self.linear(out_ch)

        return out
