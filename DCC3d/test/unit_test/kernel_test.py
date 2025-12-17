import pytest
import torch
import numpy as np
from DCC3d.src.cpu.kernel.Polynomials import HydrogenWaveFunction
from DCC3d.src.cpu.kernel.DCConv3d_kernel import DCConv3dKernelUnitPolynomials, DCConv3dKernelPolynomials


# ==========================================
# 1. 基础数学单元测试 (Polynomials.py)
# ==========================================

def test_hydrogen_init_params():
    """
    测试波函数类的参数校验逻辑
    前向传播前必须保证 N, L, M 参数合法
    """
    try:
        HydrogenWaveFunction(n=1, l=0, m=0)
        HydrogenWaveFunction(n=3, l=2, m=-1)
    except ValueError:
        pytest.fail("合法的量子数参数引发了 ValueError")

    # 非法参数测试
    with pytest.raises(ValueError, match="n must be a positive integer"):
        HydrogenWaveFunction(n=0, l=0, m=0)
    
    with pytest.raises(ValueError, match="l must be an integer between 0 and n-1"):
        HydrogenWaveFunction(n=2, l=2, m=0)

    with pytest.raises(ValueError, match="m must be an integer between -l and l"):
        HydrogenWaveFunction(n=2, l=1, m=2)

def test_hydrogen_forward_math_accuracy():
    """
    测试波函数前向计算的数学准确性
    使用物理学已知的标准值进行验证
    """
    hwf_1s = HydrogenWaveFunction(n=1, l=0, m=0)
    val_1s = hwf_1s.forward(1.0, 0.0, 0.0).item()
    expected_val = 1.0 / (np.pi ** 0.5) * np.exp(-1.0) # ≈ 0.2075537
    assert abs(val_1s - expected_val) < 1e-5

    hwf_2p = HydrogenWaveFunction(n=2, l=1, m=0)
    val_2p = hwf_2p.forward(1.0, np.pi / 2, 0.0).item()
    assert abs(val_2p - 0.0) < 1e-5

# ==========================================
# 2. 卷积核单元测试 (DCConv3d_kernel.py)
# ==========================================

def test_kernel_initialization_logic():
    """
    【关键测试】测试基函数生成的数量是否正确。
    用于捕捉循环范围 range(min(n, L)) 导致的 Bug。
    """
    N, L, M = 2, 1, 1
    # 物理推导预期数量：
    # n=1: l=0 (m=0) -> 1个
    # n=2: l=0 (m=0) -> 1个
    # n=2: l=1 (m=-1,0,1) -> 3个
    # 总计应为 5 个
    
    layer = DCConv3dKernelUnitPolynomials(N, L, M)
    
    actual_count = len(layer.polynomials)
    assert actual_count == 5, (
        f"初始化逻辑错误: 预期生成 5 个波函数, 实际生成了 {actual_count} 个。\n"
        f"提示: 请检查循环 range(..., min(n, self.L)) 是否漏掉了 l=L 的情况。"
    )

def test_kernel_forward_shape():
    """
    测试单层卷积核的输出形状 (Batch/Grid 处理能力)
    """
    OutN, n_grid = 3, 4
    layer = DCConv3dKernelUnitPolynomials(N=2, L=1, M=0)
    
    # 输入形状: (OutN, n, 3) -> (r, theta, phi)
    position = torch.rand(OutN, n_grid, 3)
    
    # 执行前向传播
    output = layer(position)
    
    # 预期输出: (OutN, n)
    assert output.shape == (OutN, n_grid)
    assert not torch.isnan(output).any(), "输出包含 NaN"

def test_kernel_forward_calculation_sum():
    """
    测试前向传播的加权求和逻辑是否正确。
    方法：将所有权重系数设为 1.0，手动计算所有波函数之和进行对比。
    """
    N, L, M = 2, 1, 0
    layer = DCConv3dKernelUnitPolynomials(N, L, M)

    for param in layer.coefficients:
        torch.nn.init.constant_(param, 1.0)

    pos_tensor = torch.tensor([[[1.0, 0.0, 0.0]]])
    
    net_output = layer(pos_tensor).item()
    num_channels = 3
    expected_sum = 0.0
    for poly in layer.polynomials:
        expected_sum += poly.forward(1.0, 0.0, 0.0).item()

    expected_sum *= num_channels 

    assert abs(net_output - expected_sum) < 1e-5

# ==========================================
# 3. 完整卷积层测试 (多通道)
# ==========================================

def test_full_layer_structure():
    """
    测试完整多通道层的输入输出维度映射
    """
    OutC, InC = 2, 3
    OutN, n_grid = 2, 5
    layer = DCConv3dKernelPolynomials(OutC, InC, N=2, L=1, M=0)
    
    position = torch.rand(OutN, n_grid, 3)
    output = layer(position)
    
    # 预期输出: (OutC, InC, OutN, n)
    assert output.shape == (OutC, InC, OutN, n_grid)