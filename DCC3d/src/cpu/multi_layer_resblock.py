import torch
import torch.nn as nn

from .flexible_dcconv3d import FlexibleDCConv3d
from .selector import SelectorConfig


class MultiLayerResBlock(nn.Module):
    """
    多层残差块 (Multi-Layer Residual Block)
    
    这是一个高级的残差连接模块，与传统的逐层残差不同，它在整个模块的输入和输出之间
    建立跨层残差连接（Skip Connection across Multiple Layers）。
    
    设计理念：
    ┌─────────────────────────────────────────┐
    │  Input (N, Cin)                         │
    │    │                                    │
    │    ├────────────────────┐  (Shortcut)   │
    │    │                    │               │
    │    ▼                    │               │
    │  Conv1 (no res, ReLU)   │               │
    │    │                    │               │
    │    ▼                    │               │
    │  Conv2 (no res, ReLU)   │               │
    │    │                    │               │
    │    ▼                    │               │
    │  Conv3 (no res, no act) │               │
    │    │                    │               │
    │    ▼                    ▼               │
    │    ├────────────────────┤               │
    │    │        Add         │               │
    │    ▼                                    │
    │   ReLU                                  │
    │    │                                    │
    │  Output (N, Cout)                       │
    └─────────────────────────────────────────┘
    
    优势：
    1. 梯度流动更顺畅：跨层残差直接连接深层和浅层，梯度可以直接回传
    2. 特征重用：保留原始输入的信息，与多层变换后的特征融合
    3. 缓解退化问题：类似 ResNet，让网络更容易学习恒等映射
    4. 激活顺序正确：先残差相加，再激活（Pre-Activation 的变体）
    
    输入和输出形状均为 (N, C)，其中：
        - N: 点的数量（可能包含合并的 Batch 维度）
        - C: 特征通道数
    """
    
    def __init__(
        self,
        channels: list[int],
        config: SelectorConfig,
        N: int,
        L: int,
        M: int,
        use_PCA: bool = True,
    ):
        """
        初始化多层残差块
        
        Args:
            channels (list[int]): 各层的通道数配置，例如 [64, 128, 256]
                - 第一个元素: 模块输入通道数 (Cin)
                - 中间元素: 各层的输出通道数
                - 最后一个元素: 模块输出通道数 (Cout)
                - 列表长度至少为 2（至少包含 1 层卷积）
                - 示例: [64, 128, 256] 会创建两层卷积 (64->128, 128->256)
            config (SelectorConfig): 邻居选择器配置
            N (int): 多项式基函数参数
            L (int): 角度基函数参数
            M (int): 径向基函数参数
            use_PCA (bool): 是否使用 PCA 进行坐标转换，默认为 True
        """
        super(MultiLayerResBlock, self).__init__()
        
        # 参数验证
        assert len(channels) >= 2, \
            f"channels 列表长度至少为 2，当前长度为 {len(channels)}"
        
        self.in_channels = channels[0]
        self.out_channels = channels[-1]
        num_layers = len(channels) - 1
        
        # 构建卷积层序列
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 判断是否为最后一层
            is_last_layer = (i == num_layers - 1)
            
            if is_last_layer:
                # 最后一层：关闭内部残差和激活
                # 原因：我们要在模块级别做残差连接和激活
                layer = FlexibleDCConv3d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    config=config,
                    N=N,
                    L=L,
                    M=M,
                    use_PCA=use_PCA,
                    use_residual=False,  # 关闭内部残差
                    activation=False,     # 关闭内部激活（关键！）
                )
            else:
                # 中间层：关闭内部残差，保留激活
                # 原因：跨层残差只在模块首尾建立，中间层需要非线性
                layer = FlexibleDCConv3d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    config=config,
                    N=N,
                    L=L,
                    M=M,
                    use_PCA=use_PCA,
                    use_residual=False,  # 关闭内部残差
                    activation=True,      # 保留激活函数
                )
            
            self.conv_layers.append(layer)
        
        # Shortcut 路径：处理输入输出通道数不匹配的情况
        if self.in_channels == self.out_channels:
            # 通道数相同，使用恒等映射（零额外参数和计算）
            self.shortcut = nn.Identity()
        else:
            # 通道数不同，使用线性投影进行维度变换
            # 将输入特征 (N, Cin) 投影到 (N, Cout)
            self.shortcut = nn.Linear(self.in_channels, self.out_channels)
        
        # 最终激活函数：在残差相加后应用
        # 这遵循 "Post-Activation" 的设计模式
        self.final_act = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        position_matrix: torch.Tensor, 
        channel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播 (Forward Pass with Multi-Layer Residual Connection)
        
        执行流程：
        1. 保存输入特征用于跨层残差连接
        2. 顺序执行所有卷积层（主路径）
        3. 计算 shortcut 路径（可能包含维度投影）
        4. 融合主路径和 shortcut 路径
        5. 应用最终激活函数
        
        Args:
            position_matrix (torch.Tensor): 点云坐标矩阵
                Shape: (N, 3), 其中 N 是点的数量，3 代表 (x, y, z)
            channel_matrix (torch.Tensor): 输入特征矩阵
                Shape: (N, Cin), 其中 Cin 是输入通道数
        
        Returns:
            output (torch.Tensor): 输出特征矩阵（经过跨层残差连接和激活）
                Shape: (N, Cout), 其中 Cout 是输出通道数
        """
        # 保存输入用于跨层残差连接
        # identity 只保存特征，不保存坐标（坐标在每层中都会使用）
        identity = channel_matrix  # Shape: (N, Cin)
        
        # 主路径：顺序执行所有卷积层
        # 每一层都会使用相同的 position_matrix 进行邻居选择
        x = channel_matrix
        for layer in self.conv_layers:
            # 注意：position_matrix 保持不变，x（特征）在逐层变换
            x = layer(position_matrix, x)  # Shape: (N, C_i) -> (N, C_{i+1})
        
        # 最后一层输出: (N, Cout)，此时还未经过激活函数
        
        # Shortcut 路径：处理输入特征的维度
        # 只处理特征，不处理坐标
        shortcut = self.shortcut(identity)  # Shape: (N, Cin) -> (N, Cout)
        
        # 跨层残差连接：融合主路径和 shortcut 路径
        # 这是关键步骤，使得梯度可以直接从输出流向输入
        x = x + shortcut  # Shape: (N, Cout)
        
        # 最终激活：在残差相加后应用
        # 确保输出的非线性特性
        output = self.final_act(x)  # Shape: (N, Cout)
        
        return output
    
    def extra_repr(self) -> str:
        """
        返回模块的额外描述信息
        """
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'num_layers={len(self.conv_layers)}'
        )


class ResidualDCConv3dNetwork(nn.Module):
    """
    基于多层残差块的点云卷积网络 (Residual DCConv3d Network)
    
    使用 MultiLayerResBlock 作为基本构建单元，堆叠构建深层点云网络。
    类似于 ResNet 的设计理念，但专门为点云数据设计。
    
    典型架构示例：
    Input (N, 64)
        ↓
    ResBlock1: [64, 64, 128]
        ↓
    ResBlock2: [128, 128, 256]
        ↓
    ResBlock3: [256, 256, 512]
        ↓
    Output (N, 512)
    
    每个 ResBlock 内部有多层卷积，但残差连接跨越整个块。
    """
    
    def __init__(
        self,
        block_configs: list[list[int]],
        config: SelectorConfig,
        N: int,
        L: int,
        M: int,
        use_PCA: bool = True,
    ):
        """
        初始化残差点云卷积网络
        
        Args:
            block_configs (list[list[int]]): 各个残差块的通道配置
                - 每个子列表定义一个 MultiLayerResBlock
                - 示例: [[64, 128, 128], [128, 256, 256], [256, 512]]
                - 会创建 3 个残差块：
                  * Block1: 64->128->128 (跨层残差: 64->128)
                  * Block2: 128->256->256 (跨层残差: 128->256)
                  * Block3: 256->512 (跨层残差: 256->512)
            config (SelectorConfig): 邻居选择器配置
            N (int): 多项式基函数参数
            L (int): 角度基函数参数
            M (int): 径向基函数参数
            use_PCA (bool): 是否使用 PCA 进行坐标转换
        """
        super(ResidualDCConv3dNetwork, self).__init__()
        
        # 验证配置连续性
        for i in range(len(block_configs) - 1):
            assert block_configs[i][-1] == block_configs[i + 1][0], \
                f"块 {i} 的输出通道 ({block_configs[i][-1]}) " \
                f"应等于块 {i+1} 的输入通道 ({block_configs[i + 1][0]})"
        
        # 构建残差块序列
        self.blocks = nn.ModuleList()
        for channels in block_configs:
            block = MultiLayerResBlock(
                channels=channels,
                config=config,
                N=N,
                L=L,
                M=M,
                use_PCA=use_PCA,
            )
            self.blocks.append(block)
    
    def forward(
        self, 
        position_matrix: torch.Tensor, 
        channel_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播（顺序执行所有残差块）
        
        Args:
            position_matrix (torch.Tensor): 点云坐标矩阵 (N, 3)
            channel_matrix (torch.Tensor): 输入特征矩阵 (N, Cin)
        
        Returns:
            output (torch.Tensor): 输出特征矩阵 (N, Cout)
        """
        x = channel_matrix
        
        # 逐块执行
        for block in self.blocks:
            x = block(position_matrix, x)
        
        return x
