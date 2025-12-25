"""
Test suite for MultiLayerResBlock
==================================

测试多层残差块及其网络架构
"""

import torch
import torch.nn as nn

from DCC3d.src.cpu.multi_layer_resblock import (
    MultiLayerResBlock,
    ResidualDCConv3dNetwork,
)
from DCC3d.src.cpu.selector import default_config


def generate_test_point_cloud(
    n_points: int = 100,
    n_channels: int = 16,
    coordinate_range: float = 10.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成用于测试的合成点云数据"""
    torch.manual_seed(seed)
    position_matrix = torch.randn(n_points, 3) * coordinate_range
    channel_matrix = torch.randn(n_points, n_channels)
    return position_matrix, channel_matrix


def test_basic_multi_layer_resblock():
    """
    测试 1: 基本多层残差块
    验证模块能够正常运行并输出正确形状
    """
    print("\n" + "=" * 60)
    print("测试 1: 基本多层残差块")
    print("=" * 60)

    n_points = 50
    channels = [64, 128, 256]  # 2 层卷积: 64->128, 128->256

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    # 创建多层残差块
    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    # 前向传播
    output = resblock(position_matrix, channel_matrix)

    # 验证输出形状
    expected_shape = (n_points, channels[-1])
    assert output.shape == expected_shape

    # 验证层数
    assert len(resblock.conv_layers) == len(channels) - 1

    # 验证 shortcut 类型（通道数不同应使用 Linear）
    assert isinstance(resblock.shortcut, nn.Linear)
    
    # 验证激活函数效果（所有输出非负）
    assert (output >= 0).all(), "最终 ReLU 后所有值应非负"

    print(f"✓ 输入通道: {channels[0]}")
    print(f"✓ 输出通道: {channels[-1]}")
    print(f"✓ 卷积层数: {len(resblock.conv_layers)}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ Shortcut 类型: {type(resblock.shortcut).__name__}")
    print(f"✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("✓ 测试通过！")


def test_identity_shortcut():
    """
    测试 2: Identity Shortcut
    验证输入输出通道数相同时使用 nn.Identity
    """
    print("\n" + "=" * 60)
    print("测试 2: Identity Shortcut（通道数相同）")
    print("=" * 60)

    n_points = 50
    channels = [128, 128, 128]  # 输入输出都是 128

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    output = resblock(position_matrix, channel_matrix)

    # 验证使用 Identity
    assert isinstance(resblock.shortcut, nn.Identity)

    print(f"✓ 输入/输出通道: {channels[0]}")
    print(f"✓ Shortcut 类型: {type(resblock.shortcut).__name__}")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！")


def test_single_layer_resblock():
    """
    测试 3: 单层残差块
    验证只有 1 层卷积时的行为
    """
    print("\n" + "=" * 60)
    print("测试 3: 单层残差块")
    print("=" * 60)

    n_points = 50
    channels = [64, 128]  # 只有 1 层卷积

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    output = resblock(position_matrix, channel_matrix)

    # 验证层数
    assert len(resblock.conv_layers) == 1

    # 验证第一层（也是最后一层）配置
    layer = resblock.conv_layers[0]
    assert layer.use_residual == False, "最后一层不应使用内部残差"
    assert layer.activation == False, "最后一层不应使用内部激活"

    print(f"✓ 卷积层数: {len(resblock.conv_layers)}")
    print(f"✓ 最后一层配置: use_residual=False, activation=False")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！")


def test_deep_resblock():
    """
    测试 4: 深层残差块
    验证多层（5 层）卷积的情况
    """
    print("\n" + "=" * 60)
    print("测试 4: 深层残差块（5 层卷积）")
    print("=" * 60)

    n_points = 50
    channels = [64, 64, 128, 128, 256, 256]  # 5 层卷积

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    output = resblock(position_matrix, channel_matrix)

    # 验证层数
    assert len(resblock.conv_layers) == 5

    # 验证中间层配置
    for i in range(4):  # 前 4 层是中间层
        layer = resblock.conv_layers[i]
        assert layer.use_residual == False, f"第 {i} 层不应使用内部残差"
        assert layer.activation == True, f"第 {i} 层应使用内部激活"

    # 验证最后一层配置
    last_layer = resblock.conv_layers[-1]
    assert last_layer.use_residual == False, "最后一层不应使用内部残差"
    assert last_layer.activation == False, "最后一层不应使用内部激活"

    print(f"✓ 卷积层数: {len(resblock.conv_layers)}")
    print(f"✓ 中间层配置: use_residual=False, activation=True")
    print(f"✓ 最后层配置: use_residual=False, activation=False")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！")


def test_gradient_flow():
    """
    测试 5: 梯度传播
    验证跨层残差连接能够改善梯度流动
    """
    print("\n" + "=" * 60)
    print("测试 5: 梯度传播")
    print("=" * 60)

    n_points = 50
    channels = [64, 128, 256]

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    # 前向传播
    output = resblock(position_matrix, channel_matrix)

    # 计算损失
    loss = output.pow(2).sum()

    # 反向传播
    loss.backward()

    # 验证梯度存在
    has_gradients = False
    for name, param in resblock.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            print(f"  ✓ {name}: 梯度范数 = {grad_norm:.6f}")

    assert has_gradients, "模型参数应该有梯度"

    print(f"✓ 损失值: {loss.item():.4f}")
    print("✓ 测试通过！")


def test_residual_dcconv3d_network():
    """
    测试 6: ResidualDCConv3dNetwork
    验证多个残差块的堆叠网络
    """
    print("\n" + "=" * 60)
    print("测试 6: ResidualDCConv3dNetwork（多块堆叠）")
    print("=" * 60)

    n_points = 50

    # 定义网络架构：3 个残差块
    block_configs = [
        [64, 64, 128],      # Block1: 64->64->128, 残差 64->128
        [128, 128, 256],    # Block2: 128->128->256, 残差 128->256
        [256, 256, 512],    # Block3: 256->256->512, 残差 256->512
    ]

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=block_configs[0][0]
    )

    # 创建网络
    network = ResidualDCConv3dNetwork(
        block_configs=block_configs,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    # 前向传播
    output = network(position_matrix, channel_matrix)

    # 验证输出形状
    expected_shape = (n_points, block_configs[-1][-1])
    assert output.shape == expected_shape

    # 验证块数
    assert len(network.blocks) == len(block_configs)

    print(f"✓ 输入通道: {block_configs[0][0]}")
    print(f"✓ 输出通道: {block_configs[-1][-1]}")
    print(f"✓ 残差块数: {len(network.blocks)}")
    print(f"✓ 输出形状: {output.shape}")
    
    # 打印每个块的配置
    for i, block in enumerate(network.blocks):
        print(f"  Block {i+1}: {block.in_channels} -> {block.out_channels}, "
              f"{len(block.conv_layers)} 层")
    
    print("✓ 测试通过！")


def test_network_gradient_flow():
    """
    测试 7: 网络级梯度传播
    验证深层网络的梯度计算
    """
    print("\n" + "=" * 60)
    print("测试 7: 网络级梯度传播")
    print("=" * 60)

    n_points = 50

    block_configs = [
        [64, 128],
        [128, 256],
        [256, 512],
    ]

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=block_configs[0][0]
    )

    network = ResidualDCConv3dNetwork(
        block_configs=block_configs,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    # 前向传播
    output = network(position_matrix, channel_matrix)

    # 计算损失
    loss = output.pow(2).sum()

    # 反向传播
    loss.backward()

    # 检查每个块的梯度
    for i, block in enumerate(network.blocks):
        has_grad = any(p.grad is not None for p in block.parameters())
        assert has_grad, f"Block {i} 应该有梯度"
        print(f"  ✓ Block {i+1}: 梯度计算正常")

    print(f"✓ 损失值: {loss.item():.4f}")
    print("✓ 测试通过！")


def test_different_point_counts():
    """
    测试 8: 不同点云大小
    验证模块能够处理不同数量的点
    """
    print("\n" + "=" * 60)
    print("测试 8: 不同点云大小")
    print("=" * 60)

    channels = [64, 128, 256]

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    # 测试不同的点数量
    point_counts = [10, 50, 100, 200]

    for n_points in point_counts:
        position_matrix, channel_matrix = generate_test_point_cloud(
            n_points=n_points, n_channels=channels[0]
        )

        output = resblock(position_matrix, channel_matrix)

        expected_shape = (n_points, channels[-1])
        assert output.shape == expected_shape

        print(f"  ✓ 点数 {n_points:3d}: 输出形状 {output.shape}")

    print("✓ 测试通过！")


def test_extra_repr():
    """
    测试 9: 模块字符串表示
    验证 extra_repr() 方法
    """
    print("\n" + "=" * 60)
    print("测试 9: 模块字符串表示")
    print("=" * 60)

    channels = [64, 128, 256, 512]

    resblock = MultiLayerResBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
    )

    model_str = str(resblock)
    print(f"✓ 模型描述:\n{model_str}")

    # 验证包含关键信息
    assert "in_channels=64" in model_str
    assert "out_channels=512" in model_str
    assert "num_layers=3" in model_str

    print("✓ 测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行 MultiLayerResBlock 测试套件")
    print("=" * 60)

    try:
        test_basic_multi_layer_resblock()
        test_identity_shortcut()
        test_single_layer_resblock()
        test_deep_resblock()
        test_gradient_flow()
        test_residual_dcconv3d_network()
        test_network_gradient_flow()
        test_different_point_counts()
        test_extra_repr()

        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！ ✓✓✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗✗✗ 测试失败！ ✗✗✗")
        print("=" * 60)
        print(f"错误信息: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
