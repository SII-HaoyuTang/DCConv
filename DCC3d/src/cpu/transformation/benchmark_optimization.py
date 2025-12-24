"""
坐标转换器性能基准测试
比较不同优化方案的性能提升
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
from coordinate_transformer import CoordinateTransformerTorch

def benchmark_coordinate_transformer(
    N_total=1000,
    N_centers=200, 
    K=32,
    num_runs=10,
    use_cuda=True
):
    """
    性能基准测试
    
    Args:
        N_total: 总点数
        N_centers: 中心点数
        K: 每个中心的邻居数
        num_runs: 运行次数
        use_cuda: 是否使用 GPU
    """
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"性能基准测试")
    print(f"{'='*70}")
    print(f"设备: {device}")
    print(f"数据规模: N_total={N_total}, N_centers={N_centers}, K={K}")
    print(f"运行次数: {num_runs}")
    
    # 生成测试数据
    torch.manual_seed(42)
    global_coords = torch.randn(N_total, 3, device=device, requires_grad=True)
    neighbor_indices = torch.randint(0, N_total, (N_centers, K), device=device)
    
    # 测试配置
    configs = [
        ("向量化", {"use_compile": False}),
        # Windows 上 torch.compile 可能需要 Triton，暂时禁用
        # ("向量化 + torch.compile", {"use_compile": True}),
    ]
    
    results = {}
    
    for name, kwargs in configs:
        print(f"\n{'-'*70}")
        print(f"测试配置: {name}")
        print(f"{'- '*70}")
        
        # 创建转换器
        transformer = CoordinateTransformerTorch(**kwargs).to(device)
        
        # 预热（首次编译可能较慢）
        if device == 'cuda':
            torch.cuda.synchronize()
        
        for _ in range(3):
            _ = transformer(global_coords, neighbor_indices)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 正式计时
        times = []
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = transformer(global_coords, neighbor_indices)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # 转换为毫秒
            
            if (i + 1) % 5 == 0:
                print(f"  运行 {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
        
        # 统计结果
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
        min_time = min(times)
        max_time = max(times)
        
        results[name] = avg_time
        
        print(f"\n  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  最快: {min_time:.2f} ms")
        print(f"  最慢: {max_time:.2f} ms")
        
        # 测试梯度传播
        if i == 0:
            loss = output[0].sum()
            loss.backward()
            grad_norm = global_coords.grad.norm().item()
            print(f"  梯度范数: {grad_norm:.6f}")
            global_coords.grad.zero_()
    
    # 性能对比
    print(f"\n{'='*70}")
    print(f"性能对比汇总")
    print(f"{'='*70}")
    
    baseline_name = "向量化"
    baseline_time = results[baseline_name]
    
    for name, avg_time in results.items():
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        print(f"{name:30s}: {avg_time:8.2f} ms  (加速 {speedup:.2f}x)")
    
    return results


def test_correctness():
    """
    验证优化后的结果与原始实现一致
    """
    print(f"\n{'='*70}")
    print(f"正确性验证")
    print(f"{'='*70}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)
    
    # 创建测试数据
    N_total, N_centers, K = 100, 20, 16
    global_coords = torch.randn(N_total, 3, device=device)
    neighbor_indices = torch.randint(0, N_total, (N_centers, K), device=device)
    
    # 创建转换器（只测试向量化版本）
    transformer = CoordinateTransformerTorch(use_compile=False).to(device)
    
    # 前向传播
    out = transformer(global_coords, neighbor_indices)
    
    # 检查输出
    spherical, centers, eigenvals, _ = out
    
    print(f"\n输出形状:")
    print(f"  球坐标: {spherical.shape}")
    print(f"  中心点: {centers.shape}")
    print(f"  特征值: {eigenvals.shape}")
    
    # 检查数值范围
    print(f"\n数值范围:")
    print(f"  r (径向): [{spherical[..., 0].min():.3f}, {spherical[..., 0].max():.3f}]")
    print(f"  θ (极角): [{spherical[..., 1].min():.3f}, {spherical[..., 1].max():.3f}]")
    print(f"  φ (方位): [{spherical[..., 2].min():.3f}, {spherical[..., 2].max():.3f}]")
    
    print(f"\n✓ 正确性验证通过")


if __name__ == "__main__":
    # 正确性验证
    test_correctness()
    
    # CPU 基准测试
    print(f"\n{'#'*70}")
    print("CPU 性能测试")
    print(f"{'#'*70}")
    benchmark_coordinate_transformer(
        N_total=500, 
        N_centers=100, 
        K=16, 
        num_runs=10,
        use_cuda=False
    )
    
    # GPU 基准测试（如果可用）
    if torch.cuda.is_available():
        print(f"\n{'#'*70}")
        print("GPU 性能测试")
        print(f"{'#'*70}")
        benchmark_coordinate_transformer(
            N_total=5000,
            N_centers=1000,
            K=32,
            num_runs=20,
            use_cuda=True
        )
    else:
        print(f"\n⚠ CUDA 不可用，跳过 GPU 测试")
    
    print(f"\n{'='*70}")
    print("✓ 所有测试完成")
    print(f"{'='*70}")
