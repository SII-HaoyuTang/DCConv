import sys
import os
import torch
import torch.nn as nn
from torch.autograd import gradcheck
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

sys.path.append(os.path.abspath("src"))

from src.cpu.kernel import polynomials_torch
from src.cpu.kernel.polynomials_torch import (
    AssociatedLaguerrePoly,
    SphericalHarmonicFunc,
    HydrogenWaveFunctionRegistry,
    HydrogenWaveFuncsSeries
)

print("Monkey patching hydrogen_wave_funcs_series to use float64...")
polynomials_torch.hydrogen_wave_funcs_series = HydrogenWaveFuncsSeries(dtype=torch.float64)
HydrogenWaveFunctionRegistry._function_cache = {}

def run_test(name, func, inputs, eps=1e-3, atol=1e-2):
    """
    它用于验证给定函数的前向计算和反向传播梯度的数值一致性，并统一处理异常和打印测试结果。
    """
    print(f"Running gradient check for: {name} ...", end=" ")
    try:
        gradcheck(func, inputs, eps=eps, atol=atol, raise_exception=True)
        print("✅ PASSED")
    except Exception as e:
        print("❌ FAILED")
        import traceback
        traceback.print_exc()

def test_laguerre():
    """
    测试连带拉盖尔多项式部分的梯度计算。
    这是氢原子波函数的径向（Radial）核心组件，验证其自动微分是否正确。
    """
    print("\n--- Test 1: Associated Laguerre Polynomials (Radial) ---")
    n, k = 4, 2
    poly = AssociatedLaguerrePoly(n, k, dtype=torch.float64)
    x = torch.rand(5, 3, dtype=torch.float64, requires_grad=True) + 0.1
    run_test("Laguerre", lambda x: poly(x), (x,))

def test_spherical_harmonics():
    """
    测试球谐函数部分的梯度计算。
    这是氢原子波函数的角向（Angular）核心组件，验证 theta 和 phi 的偏导数计算是否正确。
    """
    print("\n--- Test 2: Spherical Harmonics (Angular) ---")
    k, m = 3, 1
    sh_func = SphericalHarmonicFunc(k, m, dtype=torch.float64)
    theta = torch.rand(4, 2, dtype=torch.float64, requires_grad=True) * 3.0 + 0.1 
    phi = torch.rand(4, 2, dtype=torch.float64, requires_grad=True) * 6.0
    
    def func_wrapper(t, p): return sh_func(t, p)
    run_test(f"Spherical Harmonic", func_wrapper, (theta, phi))

def test_hydrogen_wave_full():
    """
    测试完整的氢原子波函数模型（径向 + 角向）。
    验证在标准情况（m > 0）下，从输入球坐标到最终波函数值的整个计算链路的梯度传递是否正确。
    """
    print("\n--- Test 3: Hydrogen Wave Function (Full Model) ---")
    n, l, m = 3, 2, 1
    HWFR = HydrogenWaveFunctionRegistry(n+1, l+1, m+1)
    HydrogenFuncClass = HWFR.get_function(n, l, m)
    
    r = torch.rand(2, 3, 1, dtype=torch.float32) + 0.5
    theta = torch.rand(2, 3, 1, dtype=torch.float32) * 3.0 + 0.1
    phi = torch.rand(2, 3, 1, dtype=torch.float32) * 6.0
    position = torch.cat([r, theta, phi], dim=-1).detach()
    position.requires_grad = True
    
    def forward_wrapper(pos): return HydrogenFuncClass.apply(pos)
    run_test(f"HydrogenWaveFunc", forward_wrapper, (position,))

def test_laguerre_derivative_relation():
    """
    调试测试：验证代码中使用的拉盖尔多项式导数假设是否成立。
    通过对比自动微分得到的真实导数和公式推导（下一阶多项式的负值）的预测值，检查符号或系数是否存在错误。
    """
    print("\n--- Test 4: Debugging Derivative Assumption ---")
    n, k = 4, 2
    N, K = n + k, 2 * k + 1
    
    poly = AssociatedLaguerrePoly(N, K, dtype=torch.float64)
    poly_deriv_expected = AssociatedLaguerrePoly(N, K + 1, dtype=torch.float64)
    
    x = torch.rand(5, 1, dtype=torch.float64, requires_grad=True) + 0.1
    y = poly(x)
    dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    y_deriv_pred = -poly_deriv_expected(x)
    
    mae = torch.mean(torch.abs(dydx - y_deriv_pred)).item()
    print(f"Difference betwen True Grad and Assumed Grad: {mae:.6e}")
    if mae > 1e-5:
        print("Assumption Wrong: The manual derivative implementation is incorrect.")

def test_hydrogen_coverage():
    """
    覆盖率测试：专门针对特殊量子数情况（m=0 和 m<0）进行梯度检查。
    验证代码在处理勒让德多项式路径（m=0）和复数相位/共轭路径（m<0）时的数值稳定性和逻辑正确性。
    """
    print("\n--- Test 5: Coverage Check (Negative m and Zero m) ---")
    test_cases = [(3, 1, 0), (3, 2, -1), (3, 2, -2)]
    
    for n, l, m in test_cases:
        print(f"  Testing n={n}, l={l}, m={m} ...", end=" ")
        try:
            HWFR = HydrogenWaveFunctionRegistry(n+1, l+1, abs(m)+1)
            HydrogenFuncClass = HWFR.get_function(n, l, m)
            
            position = torch.randn(2, 3, 3, dtype=torch.float64) 
            position = position + 0.5 
            position.requires_grad = True
            
            def forward_wrapper(pos): return HydrogenFuncClass.apply(pos)
            
            gradcheck(forward_wrapper, (position,), eps=1e-4, atol=1e-3)
            print("✅ PASSED")
        except Exception as e:
            print(f"❌ FAILED")
        
if __name__ == "__main__":
    print("Starting Gradient Checks...")
    torch.manual_seed(42)
    test_laguerre()
    test_spherical_harmonics()
    test_hydrogen_wave_full()
    test_laguerre_derivative_relation()
    test_hydrogen_coverage()
