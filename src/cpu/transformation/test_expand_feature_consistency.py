"""
测试 numpy 版本和 torch 版本的 expand_feature_matrix 函数一致性
"""

import numpy as np
import torch
from coordinate_transformer import CoordinateTransformer
from coordinate_transformer_torch import CoordinateTransformerTorch


def test_consistency():
    """
    验证两个版本的 expand_feature_matrix 输出一致
    """
    print("=" * 70)
    print("Numpy 版本 vs PyTorch 版本一致性测试")
    print("=" * 70)
    
    # 设置相同的随机种子
    np.random.seed(12345)
    torch.manual_seed(12345)
    
    # 创建测试数据
    N = 15
    n = 7
    Ci = 6
    
    # Numpy 数据
    data_np = np.random.randn(N, n)
    features_np = np.random.randn(N, Ci)
    
    # 转换为 PyTorch 张量
    data_torch = torch.from_numpy(data_np).float()
    features_torch = torch.from_numpy(features_np).float()
    
    print(f"\n测试数据:")
    print(f"  数据形状: ({N}, {n})")
    print(f"  特征形状: ({N}, {Ci})")
    print(f"  期望输出: ({Ci}, {N}, {n})")
    
    # Numpy 版本
    print(f"\n执行 Numpy 版本...")
    transformer_np = CoordinateTransformer()
    result_np = transformer_np.expand_feature_matrix(data_np, features_np)
    print(f"  输出形状: {result_np.shape}")
    
    # PyTorch 版本
    print(f"\n执行 PyTorch 版本...")
    transformer_torch = CoordinateTransformerTorch()
    result_torch = transformer_torch.expand_feature_matrix(data_torch, features_torch)
    print(f"  输出形状: {tuple(result_torch.shape)}")
    
    # 转换 PyTorch 结果为 numpy 进行比较
    result_torch_np = result_torch.detach().numpy()
    
    # 计算差异
    print(f"\n一致性检查:")
    diff = np.abs(result_np - result_torch_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    print(f"  相对误差: {max_diff / (np.abs(result_np).max() + 1e-10):.2e}")
    
    # 验证
    tolerance = 1e-6
    if max_diff < tolerance:
        print(f"\n✓ 一致性测试通过！(差异 < {tolerance})")
    else:
        print(f"\n✗ 一致性测试失败！(差异 = {max_diff} >= {tolerance})")
        return False
    
    # 抽样检查
    print(f"\n抽样检查 (第0个特征通道，第0个节点):")
    print(f"  Numpy:   {result_np[0, 0, :3]}")
    print(f"  PyTorch: {result_torch_np[0, 0, :3]}")
    
    print("\n" + "=" * 70)
    print("🎉 两个版本完全一致！")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    test_consistency()
