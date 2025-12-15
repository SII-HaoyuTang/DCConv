"""
坐标转换与旋转不变性模块 (Coordinate Transformation Module)
================================================================
这是整个网络的数据预处理核心


核心流程：
1. 局部格点坐标提取与中心计算
2. 相对坐标计算 (平移不变性)
3. PCA 旋转对齐 (旋转不变性)
4. 笛卡尔坐标转球极坐标

"""

import numpy as np
from typing import Tuple, Optional
from rotation_invariance import RotationInvariance


class CoordinateTransformer:
    """
    坐标转换主类
    
    将绝对坐标转换为旋转不变的球极坐标特征
    
    Attributes:
        rotation_invariance: 旋转不变性处理器
        center_method: 中心点计算方法 ('mean', 'median')
    """
    
    def __init__(self, 
                 center_method: str = 'mean',
                 use_pca: bool = True,
                 pca_stabilize: bool = True):
        """
        初始化坐标转换器
        
        Args:
            center_method: 中心点计算方法
                - 'mean': 均值中心（默认）
                - 'median': 中值中心（对异常值更鲁棒）
            use_pca: 是否使用PCA进行旋转不变性处理（默认True）
                - True: 使用PCA对齐，实现旋转不变性
                - False: 跳过PCA，直接使用相对坐标
            pca_stabilize: 是否在 PCA 中使用数值稳定化
        """
        self.center_method = center_method
        self.use_pca = use_pca
        self.rotation_invariance = RotationInvariance(stabilize=pca_stabilize)
    
    def extract_local_coordinates(self,
                                  global_coords: np.ndarray,
                                  neighbor_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        功能 1: 局部格点坐标提取与中心计算
        
        从全局坐标中提取局部邻居坐标，并计算中心点
        
        Args:
            global_coords: 全局坐标，形状 (N_total, 3)
                例如：(100, 3) 表示100个原子的绝对坐标
            neighbor_indices: 邻居索引矩阵，形状 (N_centers, K)
                例如：(100, 10) 表示100个中心点，每个有10个邻居
        
        Returns:
            local_coords: 局部坐标簇，形状 (N_centers, K, 3)
            centers: 中心点坐标，形状 (N_centers, 3)
        
        物理意义：
            - 卷积是局部操作，必须先圈定"谁和谁"一起计算
            - 中心点将作为下一层网络的输入坐标
        """
        N_centers = neighbor_indices.shape[0]  # 中心点数量
        K = neighbor_indices.shape[1]  # 每个中心的邻居数
        
        # 提取局部坐标
        # 使用高级索引：对每个中心点，取出其所有邻居的坐标
        local_coords = global_coords[neighbor_indices]  # (N_centers, K, 3)
        
        # 计算中心点坐标
        if self.center_method == 'mean':
            centers = np.mean(local_coords, axis=1)  # (N_centers, 3)
        elif self.center_method == 'median':
            centers = np.median(local_coords, axis=1)  # (N_centers, 3)
        else:
            raise ValueError(f"未知的中心计算方法: {self.center_method}")
        
        return local_coords, centers
    
    def expand_feature_matrix(self,
                              data: np.ndarray,
                              features: np.ndarray) -> np.ndarray:
        """
        特征矩阵扩展
        
        将 (N, n) 的数据和 (N, Ci) 的特征扩展为 (Ci, N, n) 的输出
        这可以用于将节点特征应用到邻居数据上，生成多通道的特征表示
        
        Args:
            data: 数据矩阵，形状 (N, n)
                例如：N 个节点，每个节点有 n 个邻居或 n 维数据
            features: 特征矩阵，形状 (N, Ci)
                例如：N 个节点，每个节点有 Ci 维特征
        
        Returns:
            expanded: 扩展后的特征矩阵，形状 (Ci, N, n)
                每个通道 i 对应特征维度 i，包含 N×n 的数据
        
        实现方式：
            使用广播机制将特征应用到数据上：
            1. data: (N, n) -> (N, 1, n)
            2. features: (N, Ci) -> (N, Ci, 1)
            3. 相乘得到: (N, Ci, n)
            4. 转置为: (Ci, N, n)
        
        示例：
            data 是 10 个节点的坐标 (10, 3)
            features 是 10 个节点的类型特征 (10, 5)
            输出是 (5, 10, 3)，表示 5 个特征通道，每个通道是 10×3 的坐标矩阵
        """
        N, n = data.shape
        N_feat, Ci = features.shape
        
        # 验证输入维度匹配
        assert N == N_feat, f"数据和特征的第一维度必须匹配: {N} != {N_feat}"
        
        # 扩展维度以进行广播
        data_expanded = data[:, np.newaxis, :]      # (N, 1, n)
        features_expanded = features[:, :, np.newaxis]  # (N, Ci, 1)
        
        # 广播相乘
        result = data_expanded * features_expanded  # (N, Ci, n)
        
        # 转置为目标形状 (Ci, N, n)
        expanded = np.transpose(result, (1, 0, 2))
        
        return expanded
    
    def compute_relative_coordinates(self,
                                     local_coords: np.ndarray,
                                     centers: np.ndarray) -> np.ndarray:
        """
        功能 2: 相对坐标计算 (Decouple)
        
        将绝对坐标转换为相对于中心点的坐标
        
        Args:
            local_coords: 局部坐标，形状 (N_centers, K, 3)
            centers: 中心点坐标，形状 (N_centers, 3)
        
        Returns:
            relative_coords: 相对坐标，形状 (N_centers, K, 3)
        
        物理意义：
            - 原子间相互作用只取决于相对距离，与绝对位置无关
            - 实现"平移不变性"：分子整体平移不影响特征
            - 公式：P_relative = P_absolute - P_center
        """
        # 广播：centers (N_centers, 3) -> (N_centers, 1, 3)
        # 然后与 local_coords (N_centers, K, 3) 相减
        relative_coords = local_coords - centers[:, np.newaxis, :]
        
        return relative_coords
    
    def apply_rotation_invariance(self,
                                  relative_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        功能 3: 实现旋转不变性 (PCA/主成分分析)
        
        
        Args:
            relative_coords: 相对坐标，形状 (N_centers, K, 3)
        
        Returns:
            aligned_coords: 对齐后的坐标，形状 (N_centers, K, 3)
            eigenvalues: 特征值，形状 (N_centers, 3)，如果不使用PCA则为零
        
        物理意义：
            - 问题：如果分子旋转，相对坐标的 (x,y,z) 值会全变
            - 解决：通过 PCA 将坐标投影到分子自身的"主轴"上
            - 结果：无论分子如何旋转，投影后的坐标都是固定的
            - 这是卷积神经网络能处理 3D 分子的核心前提
        
        算法步骤：
            1. 对每个局部点云计算协方差矩阵 (3x3)
            2. 对协方差矩阵进行对角化（特征值分解）
            3. 使用特征向量作为新坐标轴
            4. 将原始坐标投影到新坐标系

        """
        N_centers = relative_coords.shape[0]
        K = relative_coords.shape[1]
        
        # 如果不使用PCA，直接返回相对坐标
        if not self.use_pca:
            aligned_coords = relative_coords.copy()
            eigenvalues = np.zeros((N_centers, 3))
            return aligned_coords, eigenvalues
        
        aligned_coords = np.zeros_like(relative_coords)
        eigenvalues = np.zeros((N_centers, 3))
        
        # 对每个局部点云分别进行 PCA 对齐
        for i in range(N_centers):
            points = relative_coords[i]  # (K, 3)
            
            # 调用旋转不变性模块
            aligned, eigenvals, _ = self.rotation_invariance.pca_alignment(points)
            
            aligned_coords[i] = aligned
            eigenvalues[i] = eigenvals
        
        return aligned_coords, eigenvalues
    
    def cartesian_to_spherical(self, coords: np.ndarray) -> np.ndarray:
        """
        功能 4: 笛卡尔坐标转球极坐标
        
        将 (x, y, z) 转换为 (r, θ, φ)
        
        Args:
            coords: 笛卡尔坐标，形状 (..., 3)
        
        Returns:
            spherical: 球极坐标，形状 (..., 3)
                - r: 径向距离 [0, ∞)
                - θ (theta): 极角/天顶角 [0, π]
                - φ (phi): 方位角 [0, 2π)
        
        物理意义：
            - 后续卷积核基于球谐函数和连带拉盖尔多项式设计
            - 这些数学工具天然接受球坐标输入
            - 这是对接下一环节（张鑫负责的卷积核）的接口标准
        
        公式：
            r = sqrt(x² + y² + z²)
            θ = arccos(z / r)
            φ = arctan2(y, x)
        """
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        
        # 计算径向距离
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # 避免除零：当 r=0 时，角度无意义，设为 0
        epsilon = 1e-10
        r_safe = np.where(r < epsilon, epsilon, r)
        
        # 计算极角 θ (theta): [0, π]
        theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
        
        # 计算方位角 φ (phi): [0, 2π)
        # 使用 arctan2 自动处理象限
        phi = np.arctan2(y, x)
        # 将范围从 [-π, π) 转换到 [0, 2π)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        
        # 组合成球坐标
        spherical = np.stack([r, theta, phi], axis=-1)
        
        return spherical
    
    def transform(self,
                  global_coords: np.ndarray,
                  neighbor_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        完整的坐标转换流程
        
        这是主接口函数，一次性完成所有4个步骤
        
        Args:
            global_coords: 全局绝对坐标，形状 (N_total, 3)
            neighbor_indices: 邻居索引矩阵，形状 (N_centers, K)
        
        Returns:
            spherical_features: 球极坐标特征，形状 (N_centers, K, 3)
                - 每个点的特征为 (r, θ, φ)
                - 这是喂给后续卷积网络的输入
            centers: 新的中心点坐标，形状 (N_centers, 3)
                - 这是作为下一层网络的输入坐标
            eigenvalues: PCA 特征值，形状 (N_centers, 3)
                - 可选的额外特征，描述局部点云的形状
        
        数据流动过程：
            原始数据 (绝对坐标)
              ↓ [步骤1: 提取 & 中心化]
            局部点云 + 中心
              ↓ [步骤2: 相对化 - 平移不变性]
            相对坐标
              ↓ [步骤3: PCA 对齐 - 旋转不变性]
            标准姿态坐标
              ↓ [步骤4: 球坐标转换]
            球极坐标特征 → 交给卷积核
        """
        # 步骤 1: 提取局部坐标并计算中心
        local_coords, centers = self.extract_local_coordinates(
            global_coords, neighbor_indices
        )
        
        # 步骤 2: 计算相对坐标（平移不变性）
        relative_coords = self.compute_relative_coordinates(
            local_coords, centers
        )
        
        # 步骤 3: PCA 对齐（旋转不变性）
        aligned_coords, eigenvalues = self.apply_rotation_invariance(
            relative_coords
        )
        
        # 步骤 4: 转换为球极坐标
        spherical_features = self.cartesian_to_spherical(aligned_coords)
        
        return spherical_features, centers, eigenvalues
    
    def __call__(self,
                 global_coords: np.ndarray,
                 neighbor_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使类实例可调用，等同于 transform 方法
        """
        return self.transform(global_coords, neighbor_indices)


def print_transformation_info(global_coords: np.ndarray,
                             neighbor_indices: np.ndarray,
                             spherical_features: np.ndarray,
                             centers: np.ndarray,
                             eigenvalues: np.ndarray):
    """
    打印转换过程的详细信息（用于调试）
    """
    print("\n" + "=" * 70)
    print("坐标转换流程详情")
    print("=" * 70)
    
    print(f"\n输入：")
    print(f"  - 全局坐标形状: {global_coords.shape}")
    print(f"  - 邻居索引形状: {neighbor_indices.shape}")
    print(f"  - 总原子数: {global_coords.shape[0]}")
    print(f"  - 中心点数: {neighbor_indices.shape[0]}")
    print(f"  - 每个中心的邻居数: {neighbor_indices.shape[1]}")
    
    print(f"\n输出：")
    print(f"  - 球坐标特征形状: {spherical_features.shape}")
    print(f"  - 新中心点形状: {centers.shape}")
    print(f"  - PCA 特征值形状: {eigenvalues.shape}")
    
    print(f"\n第一个局部点云的球坐标特征样本：")
    print(f"  - r (径向距离): {spherical_features[0, :3, 0]}")
    print(f"  - θ (极角): {spherical_features[0, :3, 1]}")
    print(f"  - φ (方位角): {spherical_features[0, :3, 2]}")
    
    print(f"\n第一个中心点的 PCA 特征值：")
    print(f"  - 特征值: {eigenvalues[0]}")
    print(f"  - 说明: 值越大，该方向上点云越分散")
    
    print("\n" + "=" * 70)


def test_expand_feature_matrix():
    """
    测试特征矩阵扩展功能
    """
    print("\n" + "=" * 70)
    print("特征矩阵扩展测试")
    print("=" * 70)
    
    np.random.seed(999)
    
    # 创建测试数据
    N = 10      # 节点数
    n = 5       # 每个节点的数据维度（例如邻居数）
    Ci = 8      # 特征维度
    
    data = np.random.randn(N, n)
    features = np.random.randn(N, Ci)
    
    print(f"\n数据规模:")
    print(f"  数据矩阵 (data): {data.shape}")
    print(f"  特征矩阵 (features): {features.shape}")
    
    # 创建转换器
    transformer = CoordinateTransformer()
    
    # 测试特征扩展
    print(f"\n执行特征扩展:")
    expanded = transformer.expand_feature_matrix(data, features)
    
    print(f"  输入: data {data.shape} + features {features.shape}")
    print(f"  输出: expanded {expanded.shape}")
    
    # 验证形状
    expected_shape = (Ci, N, n)
    assert expanded.shape == expected_shape, f"输出形状错误: {expanded.shape} != {expected_shape}"
    
    print(f"\n✓ 输出形状验证通过: {expanded.shape}")
    
    # 数值验证
    print(f"\n数值验证:")
    # 手动计算期望结果
    data_expanded_manual = data[:, np.newaxis, :]  # (N, 1, n)
    features_expanded_manual = features[:, :, np.newaxis]  # (N, Ci, 1)
    expected_manual = np.transpose(data_expanded_manual * features_expanded_manual, (1, 0, 2))
    
    diff = np.abs(expanded - expected_manual).max()
    print(f"  与手动计算的最大差异: {diff:.2e}")
    
    assert diff < 1e-10, f"数值验证失败: {diff}"
    
    print(f"\n✓ 数值验证通过")
    
    # 测试具体数值
    print(f"\n具体数值检查:")
    print(f"  第一个特征通道的第一个节点的数据:")
    print(f"    expanded[0, 0, :] = {expanded[0, 0, :]}")
    print(f"    应该等于 data[0, :] * features[0, 0] = {data[0, :] * features[0, 0]}")
    
    assert np.allclose(expanded[0, 0, :], data[0, :] * features[0, 0]), "数值不匹配！"
    
    print(f"\n✓ 特征矩阵扩展功能正常")
    
    return True


if __name__ == "__main__":
    """
    简单测试：创建模拟数据并运行完整流程
    """
    print("=" * 70)
    print("坐标转换模块测试")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建模拟数据
    N_total = 100  # 总原子数
    N_centers = 100  # 中心点数（可以等于总原子数）
    K = 10  # 每个中心的邻居数
    
    # 生成随机的全局坐标（模拟分子结构）
    global_coords = np.random.randn(N_total, 3) * 5.0
    
    # 生成随机的邻居索引（实际应用中由选点算法给出）
    neighbor_indices = np.random.randint(0, N_total, size=(N_centers, K))
    
    print(f"\n生成模拟数据：")
    print(f"  - {N_total} 个原子")
    print(f"  - {N_centers} 个中心点")
    print(f"  - 每个中心有 {K} 个邻居")
    
    # 创建坐标转换器
    transformer = CoordinateTransformer(center_method='mean')
    
    # 执行完整转换
    print(f"\n执行坐标转换...")
    spherical_features, centers, eigenvalues = transformer(
        global_coords, neighbor_indices
    )
    
    # 打印结果信息
    print_transformation_info(
        global_coords, neighbor_indices,
        spherical_features, centers, eigenvalues
    )
    
    print("\n✓ 基本转换测试完成！")
    
    # 测试特征扩展功能
    test_expand_feature_matrix()
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！")
    print("=" * 70)
    print("=" * 70)
