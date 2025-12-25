import torch
import torch.nn as nn

from .dcconv3d import DistanceContainedConv3d
from .selector import SelectorConfig


class FlexibleDCConv3d(nn.Module):
    """
    灵活可配置的点云卷积层 (Flexible DCConv3d)
    
    封装 DistanceContainedConv3d，支持通过参数灵活控制：
    - 是否使用残差连接 (Residual Connection)
    - 是否使用激活函数 (Activation Function)
    
    这使得该算子可以适配不同的网络架构需求，例如：
    - 浅层网络可能不需要残差连接
    - 某些层可能需要线性输出（不使用激活函数）
    
    输入和输出形状均为 (N, C)，其中：
        - N: 点的数量（可能包含合并的 Batch 维度）
        - C: 特征通道数
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
        use_residual: bool = True,
        activation: bool = True,
    ):
        """
        初始化灵活可配置的点云卷积层
        
        Args:
            in_channels (int): 输入特征通道数 (Cin)
            out_channels (int): 输出特征通道数 (Cout)
            config (SelectorConfig): 邻居选择器配置
            N (int): 多项式基函数参数
            L (int): 角度基函数参数
            M (int): 径向基函数参数
            use_PCA (bool): 是否使用 PCA 进行坐标转换，默认为 True
            use_residual (bool): 是否启用残差连接，默认为 True
                - True: 添加残差路径，提升梯度流动和特征学习
                - False: 仅使用卷积路径，适合浅层网络或特定架构
            activation (bool): 是否在输出前应用激活函数（ReLU），默认为 True
                - True: 应用 ReLU 激活，引入非线性
                - False: 线性输出，适合需要保留负值或作为中间层
        """
        super(FlexibleDCConv3d, self).__init__()
        
        # 保存配置参数
        self.use_residual = use_residual
        self.activation = activation
        
        # 主卷积路径：使用 DistanceContainedConv3d
        self.conv = DistanceContainedConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            config=config,
            N=N,
            L=L,
            M=M,
            use_PCA=use_PCA,
        )
        
        # 残差路径：仅在 use_residual=True 时初始化
        if self.use_residual:
            if in_channels == out_channels:
                # 通道数相同，使用恒等映射（零开销）
                self.shortcut = nn.Identity()
            else:
                # 通道数不同，使用线性投影进行维度变换
                # Linear 适用于 (N, Cin) -> (N, Cout) 的转换
                self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            # 不使用残差连接时，设置为 None
            self.shortcut = None
        
        # 激活函数：仅在 activation=True 时初始化
        if self.activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
    
    def forward(
        self,
        position_matrix: torch.Tensor,
        channel_matrix: torch.Tensor,
        n_select: int | None = None
    ) -> torch.Tensor:
        """
        前向传播 (Forward Pass with Flexible Configuration)

        Args:
            position_matrix (torch.Tensor): 点云坐标矩阵
                Shape: (N, 3), 其中 N 是点的数量，3 代表 (x, y, z)
            channel_matrix (torch.Tensor): 点的属性特征矩阵
                Shape: (N, Cin), 其中 Cin 是输入通道数
            n_select (int | None): 从原始N个点中选择多少个点作为中心点

        Returns:
            output (torch.Tensor): 卷积后的特征（可能包含残差和激活）
                Shape: (n_select, Cout), 其中 Cout 是输出通道数
        """
        # 主路径：通过点云卷积层
        # position_matrix: (N, 3) - 用于邻居选择和坐标转换
        # channel_matrix: (N, Cin) - 输入特征
        centers, x = self.conv(position_matrix, channel_matrix, n_select)  # centers: (n_select, 3), x: (n_select, Cout)

        # 残差路径：根据配置决定是否添加
        if self.use_residual:
            # 需要先根据选择的中心点来调整残差路径
            if n_select is None or n_select >= position_matrix.shape[0]:
                # 如果选择全部点或更多，直接使用原始特征
                shortcut_input = channel_matrix
            else:
                # 如果选择了部分点，需要从原始特征中选择对应的点
                # 这里需要知道选择的点的索引，但是这个信息在 conv 中丢失了
                # 一个简单的解决方案是使用 select_n_points_minimal_variance 重新选择
                from .selector import select_n_points_minimal_variance
                _, selected_indices = select_n_points_minimal_variance(position_matrix, n_select)
                shortcut_input = channel_matrix[selected_indices]  # Shape: (n_select, Cin)

            # 计算残差路径：只处理特征，不处理坐标
            shortcut = self.shortcut(shortcut_input)  # Shape: (n_select, Cout)

            # 特征融合：主路径 + 残差路径
            x = x + shortcut  # Shape: (n_select, Cout)

        # 激活函数：根据配置决定是否应用
        if self.activation:
            x = self.act(x)  # Shape: (n_select, Cout)

        return x
    
    def extra_repr(self) -> str:
        """
        返回模块的额外描述信息，用于 print(model) 时显示配置
        """
        return (
            f'use_residual={self.use_residual}, '
            f'activation={self.activation}'
        )


class DCConv3dBlock(nn.Module):
    """
    点云卷积模块 (DCConv3d Block)
    
    包含多个 FlexibleDCConv3d 层的堆叠模块，常用于构建深层点云网络。
    支持灵活配置每一层的残差连接和激活函数。
    
    示例用法：
        # 构建一个 3 层的点云卷积模块
        block = DCConv3dBlock(
            channels=[64, 128, 128],
            config=selector_config,
            N=5, L=3, M=3,
            use_residual=[True, True, False],  # 前两层用残差，最后一层不用
            activation=[True, True, False],     # 前两层用激活，最后一层不用
        )
    """
    
    def __init__(
        self,
        channels: list[int],
        config: SelectorConfig,
        N: int,
        L: int,
        M: int,
        use_PCA: bool = True,
        use_residual: list[bool] | bool = True,
        activation: list[bool] | bool = True,
    ):
        """
        初始化点云卷积模块
        
        Args:
            channels (list[int]): 各层的通道数，例如 [64, 128, 256]
                - 第一个元素是输入通道数
                - 后续元素是各层的输出通道数
                - 总共会创建 len(channels)-1 个卷积层
            config (SelectorConfig): 邻居选择器配置
            N (int): 多项式基函数参数
            L (int): 角度基函数参数
            M (int): 径向基函数参数
            use_PCA (bool): 是否使用 PCA 进行坐标转换
            use_residual (list[bool] | bool): 每层是否使用残差连接
                - 如果是 bool：所有层使用相同配置
                - 如果是 list：为每层单独配置（长度应为 len(channels)-1）
            activation (list[bool] | bool): 每层是否使用激活函数
                - 如果是 bool：所有层使用相同配置
                - 如果是 list：为每层单独配置（长度应为 len(channels)-1）
        """
        super(DCConv3dBlock, self).__init__()
        
        num_layers = len(channels) - 1
        
        # 处理 use_residual 参数：统一转换为列表
        if isinstance(use_residual, bool):
            use_residual = [use_residual] * num_layers
        else:
            assert len(use_residual) == num_layers, \
                f"use_residual 列表长度 ({len(use_residual)}) 应等于层数 ({num_layers})"
        
        # 处理 activation 参数：统一转换为列表
        if isinstance(activation, bool):
            activation = [activation] * num_layers
        else:
            assert len(activation) == num_layers, \
                f"activation 列表长度 ({len(activation)}) 应等于层数 ({num_layers})"
        
        # 构建卷积层序列
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = FlexibleDCConv3d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                config=config,
                N=N,
                L=L,
                M=M,
                use_PCA=use_PCA,
                use_residual=use_residual[i],
                activation=activation[i],
            )
            self.layers.append(layer)
    
    def forward(
        self,
        position_matrix: torch.Tensor,
        channel_matrix: torch.Tensor,
        n_select: int | None = None
    ) -> torch.Tensor:
        """
        前向传播（顺序执行所有卷积层）

        Args:
            position_matrix (torch.Tensor): 点云坐标矩阵 (N, 3)
            channel_matrix (torch.Tensor): 输入特征矩阵 (N, Cin)
            n_select (int | None): 从原始N个点中选择多少个点作为中心点

        Returns:
            output (torch.Tensor): 输出特征矩阵 (n_select, Cout)
        """
        x = channel_matrix
        current_positions = position_matrix

        # 逐层前向传播
        for i, layer in enumerate(self.layers):
            x = layer(current_positions, x, n_select)
            # 第一层后，position_matrix 应该被更新为选择后的中心点
            # 但由于我们每层都在原始 position_matrix 上选择，这里暂时保持原样
            # 如果需要逐层传递位置信息，需要从 FlexibleDCConv3d 返回更新的位置

        return x
