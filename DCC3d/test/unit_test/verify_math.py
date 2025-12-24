import torch
import numpy as np
import math
from DCC3d.src.cpu.kernel.Polynomials import HydrogenWaveFunction as ScipyHydrogen
from DCC3d.src.cpu.kernel.polynomials_torch import HWFR

def verify_single_case(n, k, m, r_val, theta_val, phi_val, rtol=1e-5, atol=1e-6):
    """
    验证单个量子数配置 (n, k, m) 在特定坐标下的数值一致性
    注意：在 polynomials_torch 中 l (角量子数) 被命名为 k
    """
    print(f"\n{'='*20} Testing n={n}, l={k}, m={m} {'='*20}")
    
    # --------------------------
    # 1. Scipy (Reference) 计算
    # --------------------------
    # 初始化 Scipy 版本
    scipy_model = ScipyHydrogen(n=n, l=k, m=m)
    
    # 前向计算
    scipy_out = scipy_model.forward(r_val, theta_val, phi_val)
    
    # 梯度计算 (Scipy版本是分开计算三个分量的)
    scipy_grad_r = scipy_model.backward(r_val, theta_val, phi_val, 'r')
    scipy_grad_theta = scipy_model.backward(r_val, theta_val, phi_val, 'theta')
    scipy_grad_phi = scipy_model.backward(r_val, theta_val, phi_val, 'phi')
    
    scipy_grads = np.array([scipy_grad_r, scipy_grad_theta, scipy_grad_phi])

    # --------------------------
    # 2. PyTorch (Target) 计算
    # --------------------------
    # 准备输入：需要是 (Batch, Points, 3) 的形状
    # 设置 requires_grad=True 以触发 PyTorch 的梯度追踪
    position = torch.tensor([[[r_val, theta_val, phi_val]]], 
                            dtype=torch.float32, 
                            requires_grad=True)
    
    # 获取 PyTorch 的自定义 Function
    # HWFR 是一个注册表实例，我们通过 get_function 获取对应的类
    torch_func_class = HWFR.get_function(n, k, m)
    
    # 前向计算 (使用 apply 静态方法)
    torch_out = torch_func_class.apply(position)
    
    # 反向传播
    # 我们假设输出标量的梯度为 1.0，这样可以直接得到 dOutput/dInput
    torch_out.backward(torch.ones_like(torch_out))
    
    # 获取输入位置的梯度
    torch_grads = position.grad[0, 0].numpy() # 提取出 [dr, dtheta, dphi]
    
    # --------------------------
    # 3. 对比验证
    # --------------------------
    
    # --- A. 前向值对比 ---
    scipy_val = float(scipy_out)
    torch_val = float(torch_out.item())
    
    print(f"[Forward] Scipy: {scipy_val:.8f} | Torch: {torch_val:.8f}")
    
    if np.isclose(scipy_val, torch_val, rtol=rtol, atol=atol):
        print("✅ Forward Pass PASSED")
    else:
        print(f"❌ Forward Pass FAILED (Diff: {abs(scipy_val - torch_val)})")

    # --- B. 梯度对比 ---
    print(f"[Grads] Scipy (r,θ,φ): {scipy_grads}")
    print(f"[Grads] Torch (r,θ,φ): {torch_grads}")
    
    if np.allclose(scipy_grads, torch_grads, rtol=rtol, atol=atol):
        print("✅ Backward Pass PASSED")
    else:
        print(f"❌ Backward Pass FAILED")
        print(f"    Max Diff: {np.max(np.abs(scipy_grads - torch_grads))}")
def verify_batch_consistency(n, k, m):
    """
    验证单独计算和批量计算结果是否一致。
    防止广播机制 (Broadcasting) 导致维度错误。
    """
    print(f"--- Running Batch Consistency Test for n={n}, l={k}, m={m} ---")
    
    # 构造两个不同的坐标点
    p1 = [2.0, 0.5, 0.5]
    p2 = [5.0, 1.5, 2.0]
    
    # 1. 单独计算
    func_class = HWFR.get_function(n, k, m)
    
    pos1 = torch.tensor([[p1]], dtype=torch.float32, requires_grad=True)
    out1 = func_class.apply(pos1)
    out1.backward(torch.ones_like(out1))
    grad1 = pos1.grad.clone()
    
    pos2 = torch.tensor([[p2]], dtype=torch.float32, requires_grad=True)
    out2 = func_class.apply(pos2)
    out2.backward(torch.ones_like(out2))
    grad2 = pos2.grad.clone()
    
    # 2. 批量计算 (Batch size = 2)
    # 输入形状: (1, 2, 3) 模拟 1个 Batch, 2个点
    pos_batch = torch.tensor([[p1, p2]], dtype=torch.float32, requires_grad=True)
    out_batch = func_class.apply(pos_batch)
    out_batch.backward(torch.ones_like(out_batch))
    grad_batch = pos_batch.grad.clone()
    
    # 3. 对比
    # 对比值
    val_diff = torch.max(torch.abs(out_batch[0, 0] - out1[0, 0])) + \
               torch.max(torch.abs(out_batch[0, 1] - out2[0, 0]))
    
    # 对比梯度
    grad_diff = torch.max(torch.abs(grad_batch[0, 0] - grad1[0, 0])) + \
                torch.max(torch.abs(grad_batch[0, 1] - grad2[0, 0]))
                
    if val_diff < 1e-6 and grad_diff < 1e-6:
        print(f"✅ Batch Consistency PASSED (Val Diff: {val_diff:.2e}, Grad Diff: {grad_diff:.2e})")
    else:
        print(f"❌ Batch Consistency FAILED")
        print(f"Val Diff: {val_diff}")
        print(f"Grad Diff: {grad_diff}")
def main():
    # --- 基础测试坐标 ---
    r_base = 2.5
    theta_base = math.pi / 4  # 45度
    phi_base = math.pi / 3    # 60度

    print(f"{'='*20} 第一部分：基础功能测试 (原有) {'='*20}")
    # Case 1: 基态 (n=1, l=0, m=0) - 最简单情况
    verify_single_case(1, 0, 0, r_base, theta_base, phi_base)
    
    # Case 2: p轨道 (n=2, l=1, m=0) - 引入角度依赖
    verify_single_case(2, 1, 0, r_base, theta_base, phi_base)
    
    # Case 3: d轨道复数转换 (n=3, l=2, m=1) - 验证实部转换逻辑
    verify_single_case(3, 2, 1, r_base, theta_base, phi_base)

    # Case 4: d轨道负磁量子数 (n=3, l=2, m=-1) - 验证虚部转换逻辑
    verify_single_case(3, 2, -1, r_base, theta_base, phi_base)

    print(f"\n{'='*20} 第二部分：高阶量子数与特殊轨道 {'='*20}")
    # Case 5: 2s 轨道 (n=2, l=0, m=0) 
    # 测试点：径向节点。2s轨道在 r=2a0 处有一个节点，波函数变号。
    verify_single_case(2, 0, 0, r_base, theta_base, phi_base)

    # Case 6: 边界角动量 (n=4, l=3, m=3)
    # 测试点：l = n-1 且 m = l。这是该 n 下最复杂的角向分布，检查阶乘计算是否溢出或精度损失。
    verify_single_case(4, 3, 3, r_base, theta_base, phi_base)

    # Case 7: 高阶负 m (n=4, l=3, m=-3)
    # 测试点：验证 m 为负最大值时的对称性。
    verify_single_case(4, 3, -3, r_base, theta_base, phi_base)
    
    # Case 8: 高主量子数 (n=5, l=2, m=0)
    # 测试点：n=5 会涉及更高阶的拉盖尔多项式计算，验证数值稳定性。
    verify_single_case(5, 2, 0, r_base, theta_base, phi_base)

    print(f"\n{'='*20} 第三部分：极限坐标测试 {'='*20}")
    # Case 9: 极小半径 (r=0.1)
    # 测试点：接近原子核。验证 1/r 项导数计算的稳定性（避免除零或极大值）。
    verify_single_case(2, 1, 1, 0.1, theta_base, phi_base)

    # Case 10: 较大半径 (r=10.0)
    # 测试点：远离原子核。验证指数衰减项 exp(-r/n) 是否正确主导，数值应接近 0 但非 0。
    verify_single_case(2, 1, 1, 10.0, theta_base, phi_base)

    # Case 11: 特殊角度 XY平面 (theta = pi/2)
    # 测试点：cos(theta)=0。验证勒让德多项式在赤道平面上的表现。
    verify_single_case(3, 2, 1, r_base, math.pi/2, phi_base)

    # Case 12: 特殊角度 Z轴 (theta = 0, phi 任意)
    # 注意：在 theta=0 时，sin(theta)=0，导数计算可能会遇到 singularity。
    # 你的代码里有 sin_theta_judge 处理，这是一个极好的压力测试。
    # 我们稍微偏离 0 一点点 (1e-5) 来模拟数值极限，或者直接给 0 看是否由 judge 处理。
    print("\n--- Testing near Singularity (Theta close to 0) ---")
    verify_single_case(2, 1, 1, r_base, 1e-5, phi_base)
    
    print(f"\n{'='*20} 第四部分：南极点与节点测试 {'='*20}")
    # Case 13: 南极点附近的奇点 (theta -> pi)
    # 测试点：验证导数符号是否正确翻转，且没有除零错误
    print("--- Testing South Pole Singularity (Theta close to Pi) ---")
    verify_single_case(2, 1, 1, r_base, math.pi - 1e-5, phi_base)

    # Case 14: 径向节点精确位置测试 (n=2, l=0)
    # 理论上 2s 轨道的节点在 rho = 2, 即 2r/n = 2 => r = n = 2.0 (原子单位)
    # 在此位置波函数值应极接近 0
    print("--- Testing Exact Radial Node (Expect Value ~ 0) ---")
    verify_single_case(2, 0, 0, 2.0, theta_base, phi_base)

    # Case 15: 角向节点测试 (n=2, l=1, m=1) -> p_x 轨道
    # p_x 轨道正比于 sin(theta)cos(phi)。在 phi = pi/2 时应为 0。
    print("--- Testing Exact Angular Node (Expect Value ~ 0) ---")
    verify_single_case(2, 1, 1, r_base, theta_base, math.pi / 2)


    print(f"\n{'='*20} 第五部分：Batch 并行一致性测试 {'='*20}")
    verify_batch_consistency(n=3, k=2, m=1)
if __name__ == "__main__":
    main()