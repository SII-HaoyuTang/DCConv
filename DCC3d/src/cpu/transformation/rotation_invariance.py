"""
旋转不变性模块 (Rotation Invariance Module)
===================================================
通过 PCA (主成分分析) 实现旋转不变性

核心思想：
- 对局部点云计算协方差矩阵
- 通过特征值分解获得主轴方向
- 将坐标投影到主轴坐标系上
- 这样无论分子如何旋转，投影后的坐标都保持一致

"""

import numpy as np
from typing import Tuple


class RotationInvariance:
    """
    旋转不变性处理类
    
    使用 PCA（主成分分析）将局部坐标系对齐到其主轴方向，
    实现旋转不变性。
    
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
        self.stabilize = stabilize
        self.epsilon = epsilon
    
    def compute_covariance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        计算点云的协方差矩阵
        
        协方差矩阵描述了点云在各个方向上的分布特征
        
        Args:
            points: 形状为 (N, 3) 的点云坐标，已经中心化
        
        Returns:
            形状为 (3, 3) 的协方差矩阵
        """
        # 确保输入已经中心化
        assert points.shape[1] == 3, "点云必须是 (N, 3) 形状"
        
        # 协方差矩阵: C = (1/N) * P^T * P
        # 其中 P 是中心化后的坐标矩阵
        N = points.shape[0]
        cov_matrix = (points.T @ points) / N
        
        # 数值稳定化：添加小的正则化项到对角线
        if self.stabilize:
            cov_matrix += self.epsilon * np.eye(3)
        
        return cov_matrix
    
    def pca_alignment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对点云进行 PCA 对齐，实现旋转不变性
        
        步骤：
        1. 计算协方差矩阵
        2. 特征值分解（对角化）
        3. 将点云投影到主成分方向
        
        Args:
            points: 形状为 (N, 3) 的相对坐标（已中心化）
        
        Returns:
            aligned_points: 对齐后的坐标 (N, 3)
            eigenvalues: 特征值 (3,)，按降序排列
            eigenvectors: 特征向量 (3, 3)，每列是一个主成分方向
        """
        # 计算协方差矩阵
        cov_matrix = self.compute_covariance_matrix(points)
        
        # 特征值分解：对角化实对称矩阵
        # eigenvalues: 特征值（描述各主轴方向的方差）
        # eigenvectors: 特征向量（描述主轴方向）
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值从大到小排序
        # 第一主成分方差最大，包含最多信息
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 确保右手坐标系（行列式为正）
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
        
        # 将点云投影到主成分坐标系
        # 这相当于将坐标系旋转到主轴方向
        aligned_points = points @ eigenvectors
        
        return aligned_points, eigenvalues, eigenvectors
    
    def batch_pca_alignment(self, point_clouds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量处理多个点云的 PCA 对齐
        
        Args:
            point_clouds: 形状为 (batch_size, N, 3) 的点云批次
        
        Returns:
            aligned_clouds: 对齐后的点云 (batch_size, N, 3)
            eigenvalues_batch: 各点云的特征值 (batch_size, 3)
        """
        batch_size = point_clouds.shape[0]
        N = point_clouds.shape[1]
        
        aligned_clouds = np.zeros_like(point_clouds)
        eigenvalues_batch = np.zeros((batch_size, 3))
        
        for i in range(batch_size):
            aligned, eigenvals, _ = self.pca_alignment(point_clouds[i])
            aligned_clouds[i] = aligned
            eigenvalues_batch[i] = eigenvals
        
        return aligned_clouds, eigenvalues_batch
    
    def verify_rotation_invariance(self, 
                                   points: np.ndarray, 
                                   rotation_matrix: np.ndarray) -> bool:
        """
        验证旋转不变性
        
        将点云旋转后，检查 PCA 对齐结果是否一致
        
        Args:
            points: 原始点云 (N, 3)
            rotation_matrix: 旋转矩阵 (3, 3)
        
        Returns:
            是否满足旋转不变性（坐标的绝对值应该相同）
        """
        # 原始点云的 PCA 对齐
        aligned_original, _, _ = self.pca_alignment(points)
        
        # 旋转后的点云的 PCA 对齐
        rotated_points = points @ rotation_matrix.T
        aligned_rotated, _, _ = self.pca_alignment(rotated_points)
        
        # 由于主成分方向可能有符号差异，我们比较绝对值
        diff = np.abs(np.abs(aligned_original) - np.abs(aligned_rotated))
        max_diff = np.max(diff)
        
        # 数值容差
        tolerance = 1e-6
        
        return max_diff < tolerance


def generate_random_rotation_matrix() -> np.ndarray:
    """
    生成随机旋转矩阵（用于测试）
    
    使用 Gram-Schmidt 正交化过程生成正交矩阵
    
    Returns:
        3x3 正交旋转矩阵
    """
    # 生成随机矩阵
    random_matrix = np.random.randn(3, 3)
    
    # QR 分解得到正交矩阵
    Q, R = np.linalg.qr(random_matrix)
    
    # 确保行列式为 1（旋转矩阵）而非 -1（镜像）
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    return Q


if __name__ == "__main__":
    """
    简单测试：验证旋转不变性
    """
    print("=" * 60)
    print("旋转不变性模块测试")
    print("=" * 60)
    
    # 创建测试点云（10个点）
    np.random.seed(42)
    test_points = np.random.randn(10, 3)
    
    # 中心化
    test_points = test_points - test_points.mean(axis=0)
    
    # 创建旋转不变性处理器
    ri = RotationInvariance()
    
    # 1. 测试 PCA 对齐
    print("\n1. 测试 PCA 对齐")
    aligned, eigenvals, eigenvecs = ri.pca_alignment(test_points)
    print(f"   原始点云形状: {test_points.shape}")
    print(f"   对齐后形状: {aligned.shape}")
    print(f"   特征值: {eigenvals}")
    print(f"   特征向量是否正交: {np.allclose(eigenvecs.T @ eigenvecs, np.eye(3))}")
    
    # 2. 测试旋转不变性
    print("\n2. 验证旋转不变性")
    rotation = generate_random_rotation_matrix()
    is_invariant = ri.verify_rotation_invariance(test_points, rotation)
    print(f"   生成随机旋转矩阵")
    print(f"   旋转不变性验证: {'✓ 通过' if is_invariant else '✗ 失败'}")
    
    # 3. 手动验证
    print("\n3. 手动验证")
    rotated_points = test_points @ rotation.T
    aligned_original, _, _ = ri.pca_alignment(test_points)
    aligned_rotated, _, _ = ri.pca_alignment(rotated_points)
    
    print(f"   原始对齐结果样本: {aligned_original[0]}")
    print(f"   旋转对齐结果样本: {aligned_rotated[0]}")
    print(f"   差异 (应该很小): {np.max(np.abs(np.abs(aligned_original) - np.abs(aligned_rotated))):.2e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
