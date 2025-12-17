import pytest
import numpy as np
import torch
import sys
from DCC3d.src.cpu.transformation.coordinate_transformer import CoordinateTransformerTorch

@pytest.fixture
def sample_data():
    """生成标准的测试数据"""
    torch.manual_seed(42)
    N_total = 20
    N_centers = 5
    K = 4
    
    global_coords = torch.randn(N_total, 3, requires_grad=True)
    neighbor_indices = torch.randint(0, N_total, (N_centers, K))
    
    return global_coords, neighbor_indices

@pytest.fixture
def transformer():
    return CoordinateTransformerTorch(center_method='mean', use_pca=True)

# ==========================================
# 2. 不变性测试 (Invariance)
# ==========================================

def test_translation_invariance(transformer, sample_data):
    """测试平移不变性：所有点移动相同距离，输出特征应不变"""
    coords, indices = sample_data
    offset = torch.tensor([10.0, -5.0, 3.3])

    out1, _, _, _ = transformer(coords, indices)

    out2, _, _, _ = transformer(coords + offset, indices)
    
    assert torch.allclose(out1, out2, atol=1e-5), "平移不变性测试失败"

def test_rotation_invariance_eigenvalues(transformer, sample_data):
    """
    测试旋转不变性：旋转点云后，PCA 特征值应该完全不变。
    (坐标本身可能因为 PCA 轴翻转而变化，但特征值是绝对旋转不变量)
    """
    coords, indices = sample_data
    
    # 定义一个旋转矩阵 (绕 Z 轴旋转 90 度)
    theta = np.pi / 2
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_mat = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=torch.float32)
    
    coords_rotated = torch.matmul(coords, rot_mat.T).requires_grad_(True)
    _, _, eigen1, _ = transformer(coords, indices)
    _, _, eigen2, _ = transformer(coords_rotated, indices)
    
    # 验证特征值一致性
    assert torch.allclose(eigen1, eigen2, atol=1e-5), "旋转后特征值发生改变"

# ==========================================
# 3. 可微分性测试 (Differentiability)
# ==========================================

def test_gradient_flow(transformer, sample_data):
    """测试梯度是否能从输出回传到输入坐标"""
    coords, indices = sample_data
    
    # 确保清除旧梯度
    if coords.grad is not None:
        coords.grad.zero_()
        
    # 前向传播
    spherical_features, _, _, _ = transformer(coords, indices)
    
    # 定义 Loss 并反向传播
    loss = spherical_features.sum()
    loss.backward()
    
    # 验证梯度
    assert coords.grad is not None, "输入坐标没有梯度"
    assert coords.grad.abs().sum() > 0, "梯度全为 0，反向传播中断"
    assert not torch.isnan(coords.grad).any(), "梯度中出现 NaN"

# ==========================================
# 4. 数值正确性测试 (Numerical Correctness)
# ==========================================

def test_cartesian_to_spherical_logic(transformer):
    """测试笛卡尔转球坐标的数学逻辑"""
    # 构造特殊点: (1, 0, 0), (0, 1, 0), (0, 0, 1)
    # 注意 transformer 内部处理是针对 batch 的，我们这里直接测内部函数
    points = torch.tensor([
        [1.0, 0.0, 0.0], # r=1, theta=pi/2, phi=0
        [0.0, 1.0, 0.0], # r=1, theta=pi/2, phi=pi/2
        [0.0, 0.0, 1.0]  # r=1, theta=0,    phi=0 (or undefined)
    ])
    
    spherical = transformer.cartesian_to_spherical(points)
    
    # r
    assert torch.allclose(spherical[:, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(spherical[:, 1], torch.tensor([np.pi/2, np.pi/2, 0.0]), atol=1e-6)
    assert torch.allclose(spherical[0, 2], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(spherical[1, 2], torch.tensor(np.pi/2), atol=1e-6)

# ==========================================
# 5. 边界情况测试 (Edge Cases)
# ==========================================

def test_degenerate_points(transformer):
    """测试重合点（零向量）的处理，验证 epsilon 是否防止了 NaN"""
    coords = torch.zeros((2, 3), requires_grad=True)
    indices = torch.tensor([[0, 1]]) # Center 0, Neighbor 1 (都是 0,0,0)

    out, _, _, _ = transformer(coords, indices)
    
    assert not torch.isnan(out).any(), "处理零向量时产生了 NaN"
    assert torch.allclose(out[..., 0], torch.tensor(0.0), atol=2e-5)

@pytest.fixture
def transformer():
    return CoordinateTransformerTorch(center_method='mean', use_pca=True)

def create_batch(coords):
    """
    修正版辅助函数：
    CoordinateTransformerTorch 接收 (N_total, 3)，
    neighbor_indices 接收 (N_centers, K)。
    """
    N = coords.shape[0]
    indices = torch.arange(N).unsqueeze(0).long()
    
    return coords, indices

# ==========================================
# 1. 几何退化测试 (PCA Rank Deficiency)
# ==========================================

def test_pca_collinear_stability(transformer):
    """ 共线点云 (Rank=1)"""
    x = torch.linspace(0, 10, 10)
    coords = torch.zeros(10, 3)
    coords[:, 0] = x
    inputs, indices = create_batch(coords)
    _, _, eigenvalues, _ = transformer(inputs, indices)
    eig = eigenvalues[0] 
    
    assert eig[0] > 1.0, "主方向特征值应显著大于0"
    assert eig[1] < 1e-3, "第二特征值应接近0"
    assert eig[2] < 1e-3, "第三特征值应接近0"

def test_pca_coplanar_stability(transformer):
    """共面点云 (Rank=2)"""
    theta = torch.linspace(0, 2*np.pi, 20)
    coords = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=1)
    
    inputs, indices = create_batch(coords)
    _, local_coords, eigenvalues, _ = transformer(inputs, indices)
    eig = eigenvalues[0]
    assert eig[2] < 1e-3, "平面点云的最小特征值应接近 0"
    assert eig[0] > 1e-2 and eig[1] > 1e-2

# ==========================================
# 2. 特征值简并测试 (Isotropic Ambiguity)
# ==========================================
def test_pca_isotropic_cube(transformer):
    """[数学边界] 各向同性"""
    coords = torch.tensor([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=torch.float32)
    
    inputs, indices = create_batch(coords)
    _, _, eigenvalues, _ = transformer(inputs, indices)
    eig = eigenvalues[0]
    
    assert torch.allclose(eig[0], eig[1], atol=1e-4)
    assert torch.allclose(eig[1], eig[2], atol=1e-4)

# ==========================================
# 3. 球坐标奇异点测试 (Spherical Singularities)
# ==========================================

def test_spherical_poles(transformer):
    """
    [数学边界] 极点奇异性。
    点位于 Z 轴上 (0, 0, 1) 和 (0, 0, -1)。
    此时 x=0, y=0。phi = atan2(0, 0) 通常为 0。
    theta 应该是 0 或 pi。
    """
    coords = torch.tensor([
        [0.0, 0.0, 0.0], # Center
        [0.0, 0.0, 1.0], # North Pole
        [0.0, 0.0, -1.0] # South Pole
    ])
    
    spherical = transformer.cartesian_to_spherical(coords)
    assert torch.allclose(spherical[1, 1], torch.tensor(0.0), atol=1e-4), "北极 theta 应为 0"
    assert torch.allclose(spherical[2, 1], torch.tensor(np.pi), atol=1e-4), "南极 theta 应为 pi"

    assert not torch.isnan(spherical).any()

def test_spherical_origin_stability(transformer):
    """
    [数值边界] 原点/重合点。
    r -> 0。测试 epsilon 保护机制。
    """
    coords = torch.zeros((1, 3)) 
    spherical = transformer.cartesian_to_spherical(coords)

    r = spherical[0, 0]
    assert r < 1e-4, f"原点半径过大: {r}"

    assert not torch.isnan(spherical).any(), "原点导致球坐标 NaN"

# ==========================================
# 4. 尺度极端测试 (Scale Invariance)
# ==========================================

def test_micro_scale_stability(transformer):
    """[数值边界] 极小尺度输入"""
    coords = torch.rand((10, 3)) * 1e-7
    inputs, indices = create_batch(coords)
    
    _, _, eigenvalues, _ = transformer(inputs, indices)

    assert eigenvalues.max() < 1e-4, "特征值尺度应保持在低位（允许 Epsilon）"

def test_macro_scale_stability(transformer):
    """
    [数值边界] 极大尺度输入 ($10^{5}$)。
    测试 x^2 + y^2 + z^2 是否溢出，或者大数吃小数。
    """
    offset = 1e5
    coords = torch.randn((10, 3)) + offset
    inputs, indices = create_batch(coords)
    _, _, eigenvalues, _ = transformer(inputs, indices)

    assert eigenvalues.max() < 100.0, "去中心化失败，大坐标影响了 PCA 特征值"

    
if __name__ == "__main__":
    # 如果直接运行脚本，调用 pytest
    sys.exit(pytest.main(["-v", __file__]))