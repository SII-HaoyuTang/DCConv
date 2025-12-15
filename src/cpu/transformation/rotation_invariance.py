"""
旋转不变性模块 - PyTorch 可微分版本
===================================================
使用 PyTorch 实现，支持自动微分和反向传播

核心特性：
- 所有操作都是可微分的
- 支持 GPU 加速
- 可以嵌入到神经网络中进行端到端训练

注意：特征值分解是可微分的，但在特征值重复时梯度可能不稳定
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotationInvarianceTorch(nn.Module):
    """
    旋转不变性处理类 (PyTorch 可微分版本)
    
    使用 PCA（主成分分析）将局部坐标系对齐到其主轴方向，
    实现旋转不变性。所有操作支持梯度反向传播。
    
    Attributes:
        stabilize (bool): 是否使用数值稳定化技术
        epsilon (float): 用于数值稳定的极小值
    """
    
    def __init__(self, stabilize: bool = True, epsilon: float = 1e-8):
        """
        初始化旋转不变性处理器
        
        Args:
            stabilize: 是否添加正则化以提高数值稳定性
            epsilon: 正则化参数，避免协方差矩阵奇异
        """
        super().__init__()
        self.stabilize = stabilize
        self.register_buffer('epsilon', torch.tensor(epsilon))
    
    def compute_covariance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """
        计算点云的协方差矩阵（可微分）
        
        Args:
            points: 形状为 (N, 3) 的点云坐标，已经中心化
        
        Returns:
            形状为 (3, 3) 的协方差矩阵
        """
        # 协方差矩阵: C = (1/N) * P^T * P
        N = points.shape[0]
        cov_matrix = (points.T @ points) / N
        
        # 数值稳定化：添加小的正则化项到对角线
        if self.stabilize:
            cov_matrix = cov_matrix + self.epsilon * torch.eye(3, device=points.device, dtype=points.dtype)
        
        return cov_matrix
    
    def pca_alignment(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对点云进行 PCA 对齐，实现旋转不变性（可微分）
        
        步骤：
        1. 计算协方差矩阵
        2. 特征值分解（对角化）- 使用 torch.linalg.eigh（可微分）
        3. 将点云投影到主成分方向
        
        Args:
            points: 形状为 (N, 3) 的相对坐标（已中心化）
        
        Returns:
            aligned_points: 对齐后的坐标 (N, 3)
            eigenvalues: 特征值 (3,)，按降序排列
            eigenvectors: 特征向量 (3, 3)，每列是一个主成分方向
        
        注意：
            torch.linalg.eigh 是可微分的，但在特征值重复时可能数值不稳定
        """
        # 计算协方差矩阵
        cov_matrix = self.compute_covariance_matrix(points)
        
        # 特征值分解：对角化实对称矩阵（可微分操作）
        # torch.linalg.eigh 保证返回实特征值和正交特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # 按特征值从大到小排序
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 确保右手坐标系（行列式为正）
        det = torch.linalg.det(eigenvectors)
        if det < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]
        
        # 将点云投影到主成分坐标系（矩阵乘法，可微分）
        aligned_points = points @ eigenvectors
        
        return aligned_points, eigenvalues, eigenvectors
    
    def batch_pca_alignment(self, point_clouds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量处理多个点云的 PCA 对齐（可微分）
        
        Args:
            point_clouds: 形状为 (batch_size, N, 3) 的点云批次
        
        Returns:
            aligned_clouds: 对齐后的点云 (batch_size, N, 3)
            eigenvalues_batch: 各点云的特征值 (batch_size, 3)
        """
        batch_size = point_clouds.shape[0]
        N = point_clouds.shape[1]
        device = point_clouds.device
        dtype = point_clouds.dtype
        
        aligned_clouds = torch.zeros_like(point_clouds)
        eigenvalues_batch = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        
        for i in range(batch_size):
            aligned, eigenvals, _ = self.pca_alignment(point_clouds[i])
            aligned_clouds[i] = aligned
            eigenvalues_batch[i] = eigenvals
        
        return aligned_clouds, eigenvalues_batch
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：执行 PCA 对齐
        
        Args:
            points: (N, 3) 或 (batch, N, 3)
        
        Returns:
            aligned_points: 对齐后的坐标
            eigenvalues: 特征值
        """
        if points.dim() == 2:
            # 单个点云
            aligned, eigenvals, _ = self.pca_alignment(points)
            return aligned, eigenvals
        elif points.dim() == 3:
            # 批量点云
            return self.batch_pca_alignment(points)
        else:
            raise ValueError(f"输入维度必须是 2 或 3，当前为 {points.dim()}")


def generate_random_rotation_matrix_torch(device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    生成随机旋转矩阵（PyTorch 版本，用于测试）
    
    Returns:
        3x3 正交旋转矩阵
    """
    # 生成随机矩阵
    random_matrix = torch.randn(3, 3, device=device, dtype=dtype)
    
    # QR 分解得到正交矩阵
    Q, R = torch.linalg.qr(random_matrix)
    
    # 确保行列式为 1（旋转矩阵）而非 -1（镜像）
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q


if __name__ == "__main__":
    """
    测试可微分性
    """
    print("=" * 70)
    print("旋转不变性模块 - PyTorch 可微分版本测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建测试点云（需要梯度）
    torch.manual_seed(42)
    test_points = torch.randn(10, 3, device=device, requires_grad=True)
    
    # 中心化（保持梯度）
    test_points_centered = test_points - test_points.mean(dim=0, keepdim=True)
    
    # 创建旋转不变性处理器
    ri = RotationInvarianceTorch().to(device)
    
    # 1. 前向传播
    print("\n1. 前向传播测试")
    aligned, eigenvals = ri(test_points_centered)
    print(f"   输入形状: {test_points_centered.shape}")
    print(f"   输出形状: {aligned.shape}")
    print(f"   特征值: {eigenvals.detach().cpu().numpy()}")
    print(f"   输入需要梯度: {test_points.requires_grad}")
    print(f"   输出需要梯度: {aligned.requires_grad}")
    
    # 2. 反向传播测试
    print("\n2. 反向传播测试")
    
    # 定义一个简单的损失：对齐后坐标的平方和
    loss = (aligned ** 2).sum()
    print(f"   损失值: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    print(f"   ✓ 反向传播成功")
    print(f"   输入梯度形状: {test_points.grad.shape}")
    print(f"   输入梯度范数: {test_points.grad.norm().item():.6f}")
    print(f"   梯度样本: {test_points.grad[0].detach().cpu().numpy()}")
    
    # 3. 批量处理测试
    print("\n3. 批量处理测试")
    batch_points = torch.randn(5, 10, 3, device=device, requires_grad=True)
    batch_aligned, batch_eigenvals = ri(batch_points)
    
    print(f"   批量输入形状: {batch_points.shape}")
    print(f"   批量输出形状: {batch_aligned.shape}")
    print(f"   批量特征值形状: {batch_eigenvals.shape}")
    
    batch_loss = (batch_aligned ** 2).sum()
    batch_loss.backward()
    print(f"   ✓ 批量反向传播成功")
    print(f"   批量梯度范数: {batch_points.grad.norm().item():.6f}")
    
    # 4. 旋转不变性验证
    print("\n4. 旋转不变性验证")
    test_points_np = torch.randn(10, 3, device=device)
    test_points_np = test_points_np - test_points_np.mean(dim=0)
    
    # 原始 PCA
    aligned_orig, _, _ = ri.pca_alignment(test_points_np)
    
    # 旋转后 PCA
    rotation = generate_random_rotation_matrix_torch(device=device)
    rotated_points = test_points_np @ rotation.T
    aligned_rot, _, _ = ri.pca_alignment(rotated_points)
    
    # 比较（取绝对值）
    diff = torch.abs(torch.abs(aligned_orig) - torch.abs(aligned_rot))
    max_diff = diff.max().item()
    
    print(f"   最大差异: {max_diff:.2e}")
    print(f"   旋转不变性: {'✓ 通过' if max_diff < 1e-5 else '✗ 失败'}")
    
    print("\n" + "=" * 70)
    print("✓ PyTorch 可微分版本测试完成！")
    print("=" * 70)
