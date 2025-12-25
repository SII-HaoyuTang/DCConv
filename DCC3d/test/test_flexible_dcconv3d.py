"""
Test suite for FlexibleDCConv3d
================================

测试灵活可配置的点云卷积层 FlexibleDCConv3d 和 DCConv3dBlock
"""

import torch
import torch.nn as nn

from DCC3d.src.cpu.flexible_dcconv3d import FlexibleDCConv3d, DCConv3dBlock
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


def test_with_residual_and_activation():
    """
    测试 1: 完整配置（残差 + 激活）
    验证默认配置下的完整功能
    """
    print("\n" + "=" * 60)
    print("测试 1: 完整配置（use_residual=True, activation=True）")
    print("=" * 60)

    n_points = 50
    in_channels = 16
    out_channels = 32

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=in_channels
    )

    model = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=True,   # 启用残差
        activation=True,      # 启用激活
    )

    output = model(position_matrix, channel_matrix)

    # 验证输出形状
    assert output.shape == (n_points, out_channels)
    
    # 验证残差路径存在
    assert model.shortcut is not None, "应该初始化残差路径"
    assert isinstance(model.shortcut, nn.Linear), "通道数不同应使用 Linear"
    
    # 验证激活函数存在
    assert model.act is not None, "应该初始化激活函数"
    
    # 验证 ReLU 效果（所有输出非负）
    assert (output >= 0).all(), "ReLU 后所有值应非负"

    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 残差路径: {type(model.shortcut).__name__}")
    print(f"✓ 激活函数: {type(model.act).__name__}")
    print(f"✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("✓ 测试通过！")


def test_without_residual():
    """
    测试 2: 无残差连接（仅卷积 + 激活）
    验证 use_residual=False 时的行为
    """
    print("\n" + "=" * 60)
    print("测试 2: 无残差配置（use_residual=False, activation=True）")
    print("=" * 60)

    n_points = 50
    in_channels = 16
    out_channels = 32

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=in_channels
    )

    model = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=False,  # 禁用残差
        activation=True,      # 启用激活
    )

    output = model(position_matrix, channel_matrix)

    # 验证输出形状
    assert output.shape == (n_points, out_channels)
    
    # 验证残差路径不存在
    assert model.shortcut is None, "不应该初始化残差路径"
    
    # 验证激活函数存在
    assert model.act is not None, "应该初始化激活函数"
    
    # 验证 ReLU 效果
    assert (output >= 0).all(), "ReLU 后所有值应非负"

    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 残差路径: {model.shortcut}")
    print(f"✓ 激活函数: {type(model.act).__name__}")
    print("✓ 测试通过！")


def test_without_activation():
    """
    测试 3: 无激活函数（残差 + 线性输出）
    验证 activation=False 时的行为
    """
    print("\n" + "=" * 60)
    print("测试 3: 无激活配置（use_residual=True, activation=False）")
    print("=" * 60)

    n_points = 50
    in_channels = 16
    out_channels = 32

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=in_channels
    )

    model = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=True,   # 启用残差
        activation=False,     # 禁用激活
    )

    output = model(position_matrix, channel_matrix)

    # 验证输出形状
    assert output.shape == (n_points, out_channels)
    
    # 验证残差路径存在
    assert model.shortcut is not None, "应该初始化残差路径"
    
    # 验证激活函数不存在
    assert model.act is None, "不应该初始化激活函数"
    
    # 验证可能包含负值（无 ReLU）
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 残差路径: {type(model.shortcut).__name__}")
    print(f"✓ 激活函数: {model.act}")
    print(f"✓ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"✓ 包含负值: {(output < 0).any().item()} (无激活函数)")
    print("✓ 测试通过！")


def test_minimal_configuration():
    """
    测试 4: 最小配置（仅卷积）
    验证 use_residual=False, activation=False 时的行为
    """
    print("\n" + "=" * 60)
    print("测试 4: 最小配置（use_residual=False, activation=False）")
    print("=" * 60)

    n_points = 50
    in_channels = 16
    out_channels = 32

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=in_channels
    )

    model = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=False,  # 禁用残差
        activation=False,     # 禁用激活
    )

    output = model(position_matrix, channel_matrix)

    # 验证输出形状
    assert output.shape == (n_points, out_channels)
    
    # 验证残差和激活都不存在
    assert model.shortcut is None, "不应该初始化残差路径"
    assert model.act is None, "不应该初始化激活函数"

    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 残差路径: {model.shortcut}")
    print(f"✓ 激活函数: {model.act}")
    print(f"✓ 这是最简化的配置，仅包含卷积计算")
    print("✓ 测试通过！")


def test_identity_shortcut():
    """
    测试 5: Identity 残差路径
    验证通道数相同时使用 nn.Identity()
    """
    print("\n" + "=" * 60)
    print("测试 5: Identity 残差路径（in_channels == out_channels）")
    print("=" * 60)

    n_points = 50
    channels = 32  # 输入输出通道数相同

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels
    )

    model = FlexibleDCConv3d(
        in_channels=channels,
        out_channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=True,
        activation=True,
    )

    output = model(position_matrix, channel_matrix)

    # 验证使用 Identity
    assert isinstance(model.shortcut, nn.Identity), "通道数相同应使用 Identity"

    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 残差路径: {type(model.shortcut).__name__}")
    print("✓ 测试通过！")


def test_dcconv3d_block():
    """
    测试 6: DCConv3dBlock（多层堆叠）
    验证模块化堆叠功能
    """
    print("\n" + "=" * 60)
    print("测试 6: DCConv3dBlock（多层堆叠）")
    print("=" * 60)

    n_points = 50
    channels = [16, 32, 64, 128]  # 4 层：16->32->64->128

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    # 创建块：前两层用残差和激活，最后一层不用激活
    block = DCConv3dBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=[True, True, True],      # 所有层都用残差
        activation=[True, True, False],       # 最后一层不用激活
    )

    output = block(position_matrix, channel_matrix)

    # 验证输出形状
    expected_shape = (n_points, channels[-1])
    assert output.shape == expected_shape

    # 验证层数
    assert len(block.layers) == len(channels) - 1

    # 验证各层配置
    assert block.layers[0].use_residual == True
    assert block.layers[0].activation == True
    
    assert block.layers[1].use_residual == True
    assert block.layers[1].activation == True
    
    assert block.layers[2].use_residual == True
    assert block.layers[2].activation == False

    print(f"✓ 输入通道: {channels[0]}")
    print(f"✓ 输出通道: {channels[-1]}")
    print(f"✓ 层数: {len(block.layers)}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ 各层配置验证通过")
    print("✓ 测试通过！")


def test_dcconv3d_block_uniform_config():
    """
    测试 7: DCConv3dBlock（统一配置）
    验证使用 bool 参数统一配置所有层
    """
    print("\n" + "=" * 60)
    print("测试 7: DCConv3dBlock（统一配置）")
    print("=" * 60)

    n_points = 50
    channels = [16, 32, 64]  # 2 层

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=channels[0]
    )

    # 使用 bool 参数，所有层统一配置
    block = DCConv3dBlock(
        channels=channels,
        config=default_config,
        N=5, L=3, M=3,
        use_PCA=True,
        use_residual=True,   # 所有层都启用
        activation=True,      # 所有层都启用
    )

    output = block(position_matrix, channel_matrix)

    # 验证所有层配置相同
    for i, layer in enumerate(block.layers):
        assert layer.use_residual == True, f"第 {i} 层 use_residual 应为 True"
        assert layer.activation == True, f"第 {i} 层 activation 应为 True"

    print(f"✓ 所有层统一配置: use_residual=True, activation=True")
    print(f"✓ 输出形状: {output.shape}")
    print("✓ 测试通过！")


def test_gradient_flow():
    """
    测试 8: 梯度传播
    验证不同配置下的梯度计算
    """
    print("\n" + "=" * 60)
    print("测试 8: 梯度传播")
    print("=" * 60)

    n_points = 50
    in_channels = 16
    out_channels = 32

    position_matrix, channel_matrix = generate_test_point_cloud(
        n_points=n_points, n_channels=in_channels
    )

    # 测试有残差的情况
    model_with_residual = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_residual=True,
        activation=True,
    )

    output = model_with_residual(position_matrix, channel_matrix)
    loss = output.pow(2).sum()
    loss.backward()

    # 检查梯度
    has_grad = any(p.grad is not None for p in model_with_residual.parameters())
    assert has_grad, "模型应该有梯度"

    print(f"✓ 有残差连接 - 梯度计算正常")
    print(f"✓ 损失值: {loss.item():.4f}")

    # 测试无残差的情况
    model_without_residual = FlexibleDCConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        config=default_config,
        N=5, L=3, M=3,
        use_residual=False,
        activation=True,
    )

    output = model_without_residual(position_matrix, channel_matrix)
    loss = output.pow(2).sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in model_without_residual.parameters())
    assert has_grad, "模型应该有梯度"

    print(f"✓ 无残差连接 - 梯度计算正常")
    print(f"✓ 损失值: {loss.item():.4f}")
    print("✓ 测试通过！")


def test_extra_repr():
    """
    测试 9: 模型字符串表示
    验证 extra_repr() 方法
    """
    print("\n" + "=" * 60)
    print("测试 9: 模型字符串表示")
    print("=" * 60)

    model = FlexibleDCConv3d(
        in_channels=16,
        out_channels=32,
        config=default_config,
        N=5, L=3, M=3,
        use_residual=True,
        activation=False,
    )

    model_str = str(model)
    print(f"✓ 模型描述:\n{model_str}")
    
    # 验证包含配置信息
    assert "use_residual=True" in model_str
    assert "activation=False" in model_str
    
    print("✓ 测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行 FlexibleDCConv3d 测试套件")
    print("=" * 60)

    try:
        test_with_residual_and_activation()
        test_without_residual()
        test_without_activation()
        test_minimal_configuration()
        test_identity_shortcut()
        test_dcconv3d_block()
        test_dcconv3d_block_uniform_config()
        test_gradient_flow()
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
