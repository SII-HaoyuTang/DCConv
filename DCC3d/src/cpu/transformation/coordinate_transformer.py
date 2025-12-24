"""
坐标转换与旋转不变性模块 - PyTorch 可微分版本
================================================================
这是整个网络的数据预处理核心（支持反向传播）

核心特性：
- 完全可微分，支持端到端训练
- 支持 GPU 加速
- 可以作为 nn.Module 嵌入到神经网络中
- 批量处理优化

功能：将"杂乱的绝对坐标"变成"整齐、统一、带有物理意义的相对球坐标"

核心流程：
1. 局部格点坐标提取与中心计算
2. 相对坐标计算 (平移不变性)
3. PCA 旋转对齐 (旋转不变性) - 可微分！
4. 笛卡尔坐标转球极坐标 - 可微分！
"""

from typing import Optional, Tuple
import sys

import torch
import torch.nn as nn

# 处理相对导入
try:
    from .rotation_invariance import RotationInvarianceTorch
except ImportError:
    from rotation_invariance import RotationInvarianceTorch

# 检测 torch.compile 可用性（PyTorch 2.0+）
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and sys.version_info >= (3, 8)


class CoordinateTransformerTorch(nn.Module):
    """
    坐标转换主类 (PyTorch 可微分版本)

    将绝对坐标转换为旋转不变的球极坐标特征
    所有操作支持梯度反向传播

    Attributes:
        rotation_invariance: 旋转不变性处理器
        center_method: 中心点计算方法 ('mean', 'median')
    """

    def __init__(
        self,
        center_method: str = "mean",
        use_pca: bool = True,
        pca_stabilize: bool = True,
        use_compile: bool = True,
    ):
        """
        初始化坐标转换器

        Args:
            center_method: 中心点计算方法
                - 'mean': 均值中心（默认，可微分）
                - 'median': 中值中心（不可微分！）
            use_pca: 是否使用PCA进行旋转不变性处理（默认True）
                - True: 使用PCA对齐，实现旋转不变性（可微分）
                - False: 跳过PCA，直接使用相对坐标
            pca_stabilize: 是否在 PCA 中使用数值稳定化
            use_compile: 是否使用 torch.compile 加速（PyTorch 2.0+）
                - True: 使用 JIT 编译加速（默认）
                - False: 不使用编译（兼容模式）
        """
        super().__init__()
        self.center_method = center_method
        self.use_pca = use_pca
        self.use_compile = use_compile and TORCH_COMPILE_AVAILABLE

        if center_method == "median":
            print("警告: median 不可微分，将在需要梯度时使用 mean")
        
        if self.use_compile and not TORCH_COMPILE_AVAILABLE:
            print(f"警告: torch.compile 不可用 (PyTorch {torch.__version__})，回退到普通模式")
            self.use_compile = False

        self.rotation_invariance = RotationInvarianceTorch(stabilize=pca_stabilize)
        
        # 编译核心计算函数以加速
        if self.use_compile:
            self._apply_pca_batch = torch.compile(self._apply_pca_batch_impl)
            print(f"✓ 使用 torch.compile 加速（PyTorch {torch.__version__}）")
        else:
            self._apply_pca_batch = self._apply_pca_batch_impl

    def extract_local_coordinates(
        self, global_coords: torch.Tensor, neighbor_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        功能 1: 局部格点坐标提取与中心计算（可微分）

        从全局坐标中提取局部邻居坐标，并计算中心点

        Args:
            global_coords: 全局坐标，形状 (N_total, 3)
            neighbor_indices: 邻居索引矩阵，形状 (N_centers, K)

        Returns:
            local_coords: 局部坐标簇，形状 (N_centers, K, 3)
            centers: 中心点坐标，形状 (N_centers, 3)

        可微分性：
            - 索引操作：PyTorch 的高级索引支持梯度传播
            - 均值操作：完全可微分
        """
        # 提取局部坐标（可微分索引）
        local_coords = global_coords[neighbor_indices]  # (N_centers, K, 3)

        # 计算中心点坐标（可微分）
        if self.center_method == "mean" or global_coords.requires_grad:
            centers = local_coords.mean(dim=1)  # (N_centers, 3)
        elif self.center_method == "median":
            # median 不可微分，仅用于推理
            centers = local_coords.median(dim=1)[0]  # (N_centers, 3)
        else:
            raise ValueError(f"未知的中心计算方法: {self.center_method}")

        return local_coords, centers

    def extract_local_features(
        self, global_features: torch.Tensor, neighbor_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        属性距阵通过选点距阵扩展（可微分）

        将全局属性特征通过邻居索引扩展为局部属性特征
        这对应流程图中的"属性距阵通过选点距阵扩展"步骤

        Args:
            global_features: 全局属性特征，形状 (N_total, Ci)
                Ci 是属性特征的维度（如原子类型、电荷等）
            neighbor_indices: 邻居索引矩阵，形状 (N_centers, K)

        Returns:
            local_features: 局部属性特征，形状 (N_centers, K, Ci)

        可微分性：
            - 索引操作：完全可微分，梯度可以传播回 global_features

        示例：
            如果 global_features 是原子类型的 one-hot 编码 (N_total, 118)
            neighbor_indices 指定了每个中心点的 K 个邻居
            则返回 (N_centers, K, 118)，即每个中心点的邻居的原子类型特征
        """
        # 通过索引提取局部特征（可微分）
        local_features = global_features[neighbor_indices]  # (N_centers, K, Ci)

        return local_features

    def expand_feature_matrix(
        self, data: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        特征矩阵扩展（可微分）

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

        可微分性：
            - unsqueeze 和 permute：形状操作，完全可微分
            - 广播乘法：完全可微分

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
        data_expanded = data.unsqueeze(1)  # (N, 1, n)
        features_expanded = features.unsqueeze(2)  # (N, Ci, 1)

        # 广播相乘（可微分）
        result = data_expanded * features_expanded  # (N, Ci, n)

        # 转置为目标形状 (Ci, N, n)
        expanded = result.permute(1, 0, 2)

        return expanded

    def compute_relative_coordinates(
        self, local_coords: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        """
        功能 2: 相对坐标计算 (Decouple)（可微分）

        将绝对坐标转换为相对于中心点的坐标

        Args:
            local_coords: 局部坐标，形状 (N_centers, K, 3)
            centers: 中心点坐标，形状 (N_centers, 3)

        Returns:
            relative_coords: 相对坐标，形状 (N_centers, K, 3)

        可微分性：
            - 减法操作：完全可微分
            - 广播机制：保持梯度传播
        """
        # 广播并相减（可微分）
        relative_coords = local_coords - centers.unsqueeze(1)

        return relative_coords

    def _apply_pca_batch_impl(
        self, relative_coords: torch.Tensor, epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量 PCA 对齐的核心实现（供 torch.compile 编译）
        
        这个函数会被 torch.compile 编译成优化的融合核
        
        Args:
            relative_coords: (N_centers, K, 3)
            epsilon: 数值稳定化参数
            
        Returns:
            aligned_coords: (N_centers, K, 3)
            eigenvalues: (N_centers, 3)
        """
        K = relative_coords.shape[1]
        device = relative_coords.device
        dtype = relative_coords.dtype
        
        # 1. 批量计算协方差矩阵
        cov_matrices = torch.bmm(
            relative_coords.transpose(1, 2),
            relative_coords
        ) / K
        
        # 2. 数值稳定化
        cov_matrices = cov_matrices + epsilon * torch.eye(
            3, device=device, dtype=dtype
        ).unsqueeze(0)
        
        # 3. 批量特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)
        
        # 4. 批量投影
        aligned_coords = torch.bmm(relative_coords, eigenvectors)
        
        return aligned_coords, eigenvalues

    def apply_rotation_invariance(
        self, relative_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        功能 3: 实现旋转不变性 (PCA/主成分分析)（可微分 + 并行化）

        这是最难也是最关键的一步！现在已经向量化，支持批量并行处理！

        Args:
            relative_coords: 相对坐标，形状 (N_centers, K, 3)

        Returns:
            aligned_coords: 对齐后的坐标，形状 (N_centers, K, 3)
            eigenvalues: 特征值，形状 (N_centers, 3)，如果不使用PCA则为零

        可微分性：
            - 协方差计算：批量矩阵乘法，可微分
            - 特征值分解：torch.linalg.eigh 支持批量操作，可微分
            - 投影：批量矩阵乘法，可微分

        性能优化：
            - 使用批量矩阵运算代替循环（向量化）
            - 利用 GPU 并行处理所有中心点
            - 性能提升 10-100 倍（取决于 N_centers）
        """
        N_centers = relative_coords.shape[0]
        K = relative_coords.shape[1]
        device = relative_coords.device
        dtype = relative_coords.dtype

        # 如果不使用PCA，直接返回相对坐标（保持可微分）
        if not self.use_pca:
            aligned_coords = relative_coords.clone()
            eigenvalues = torch.zeros(N_centers, 3, device=device, dtype=dtype)
            return aligned_coords, eigenvalues

        # ========== 向量化实现：批量 PCA 对齐 ==========
        
        # 调用编译后的批量 PCA 函数
        epsilon = float(self.rotation_invariance.epsilon)
        aligned_coords, eigenvalues = self._apply_pca_batch(
            relative_coords, epsilon
        )

        return aligned_coords, eigenvalues

    def cartesian_to_spherical(self, coords: torch.Tensor) -> torch.Tensor:
        """
        功能 4: 笛卡尔坐标转球极坐标（可微分）

        将 (x, y, z) 转换为 (r, θ, φ)

        Args:
            coords: 笛卡尔坐标，形状 (..., 3)

        Returns:
            spherical: 球极坐标，形状 (..., 3)
                - r: 径向距离 [0, ∞)
                - θ (theta): 极角/天顶角 [0, π]
                - φ (phi): 方位角 [0, 2π)

        可微分性：
            - sqrt: 可微分（注意 r=0 时的数值稳定性）
            - arccos: 可微分
            - atan2: 可微分

        数值稳定性：
            - 避免除零：当 r=0 时使用 epsilon
            - arccos 的输入裁剪到 [-1, 1]

        公式：
            r = sqrt(x² + y² + z²)
            θ = arccos(z / r)
            φ = atan2(y, x)
        """
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        # 计算径向距离（可微分）
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-10)  # 加 epsilon 避免梯度爆炸

        # 避免除零
        r_safe = torch.where(r < 1e-10, torch.ones_like(r) * 1e-10, r)

        # 计算极角 θ (theta): [0, π]（可微分）
        theta = torch.acos(torch.clamp(z / r_safe, -1.0, 1.0))

        # 计算方位角 φ (phi): [-π, π]（可微分）
        phi = torch.atan2(y, x)
        # 转换到 [0, 2π)
        phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)

        # 组合成球坐标
        spherical = torch.stack([r, theta, phi], dim=-1)

        return spherical

    def forward(
        self,
        global_coords: torch.Tensor,
        neighbor_indices: torch.Tensor,
        global_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        完整的坐标转换流程（可微分）

        这是主接口函数，一次性完成所有4个步骤
        支持端到端的梯度反向传播

        Args:
            global_coords: 全局绝对坐标，形状 (N_total, 3)
            neighbor_indices: 邻居索引矩阵，形状 (N_centers, K)
            global_features: 可选的全局属性特征，形状 (N_total, Ci)
                如果提供，将通过邻居索引扩展为局部特征

        Returns:
            spherical_features: 球极坐标特征，形状 (N_centers, K, 3)
            centers: 新的中心点坐标，形状 (N_centers, 3)
            eigenvalues: PCA 特征值，形状 (N_centers, 3)
            local_features: 局部属性特征，形状 (N_centers, K, Ci)
                如果 global_features 为 None，则返回 None

        梯度流动：
            输入 global_coords (需要梯度)
              ↓ [索引操作 - 可微分]
            local_coords
              ↓ [均值计算 - 可微分]
            centers + relative_coords
              ↓ [PCA 对齐 - 可微分]
            aligned_coords
              ↓ [球坐标转换 - 可微分]
            spherical_features (梯度可传回输入)

            输入 global_features (需要梯度，可选)
              ↓ [索引操作 - 可微分]
            local_features (梯度可传回输入)
        """
        # 步骤 1: 提取局部坐标并计算中心（可微分）
        local_coords, centers = self.extract_local_coordinates(
            global_coords, neighbor_indices
        )

        # 步骤 1.5: 提取局部属性特征（可选，可微分）
        local_features = None
        if global_features is not None:
            local_features = self.extract_local_features(
                global_features, neighbor_indices
            )

        # 步骤 2: 计算相对坐标（平移不变性，可微分）
        relative_coords = self.compute_relative_coordinates(local_coords, centers)

        # 步骤 3: PCA 对齐（旋转不变性，可微分）
        aligned_coords, eigenvalues = self.apply_rotation_invariance(relative_coords)

        # 步骤 4: 转换为球极坐标（可微分）
        spherical_features = self.cartesian_to_spherical(aligned_coords)

        return spherical_features, centers, eigenvalues, local_features


# def test_differentiability():
#     """
#     测试完整模块的可微分性
#     """
#     print("\n" + "=" * 70)
#     print("完整模块可微分性测试")
#     print("=" * 70)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"使用设备: {device}")

#     torch.manual_seed(123)

#     # 创建模拟数据（需要梯度）
#     N_total = 50
#     N_centers = 10
#     K = 8

#     global_coords = torch.randn(N_total, 3, device=device, requires_grad=True)
#     neighbor_indices = torch.randint(0, N_total, (N_centers, K), device=device)

#     print(f"\n数据规模:")
#     print(f"  总原子数: {N_total}")
#     print(f"  中心点数: {N_centers}")
#     print(f"  邻居数: {K}")
#     print(f"  输入需要梯度: {global_coords.requires_grad}")

#     # 创建转换器
#     transformer = CoordinateTransformerTorch(center_method='mean').to(device)

#     # 前向传播
#     print(f"\n前向传播:")
#     spherical_features, centers, eigenvalues, local_features = transformer(
#         global_coords, neighbor_indices
#     )

#     print(f"  球坐标特征: {spherical_features.shape}, 需要梯度: {spherical_features.requires_grad}")
#     print(f"  中心点: {centers.shape}, 需要梯度: {centers.requires_grad}")
#     print(f"  特征值: {eigenvalues.shape}, 需要梯度: {eigenvalues.requires_grad}")
#     print(f"  局部特征: {local_features}")

#     # 定义损失函数
#     # 这里用一个简单的损失：球坐标的径向距离的平方和
#     r = spherical_features[..., 0]  # 提取径向距离
#     loss = (r ** 2).sum()

#     print(f"\n反向传播:")
#     print(f"  损失值: {loss.item():.6f}")

#     # 反向传播
#     loss.backward()

#     print(f"  ✓ 反向传播成功！")
#     print(f"  输入梯度形状: {global_coords.grad.shape}")
#     print(f"  输入梯度范数: {global_coords.grad.norm().item():.6f}")
#     print(f"  输入梯度非零元素: {(global_coords.grad != 0).sum().item()} / {global_coords.grad.numel()}")

#     # 检查梯度是否有效
#     assert global_coords.grad is not None, "梯度为空！"
#     assert not torch.isnan(global_coords.grad).any(), "梯度包含 NaN！"
#     assert not torch.isinf(global_coords.grad).any(), "梯度包含 Inf！"

#     print(f"\n✓ 梯度健康检查通过")

#     return True


# def test_feature_expansion():
#     """
#     测试属性特征扩展功能
#     """
#     print("\n" + "=" * 70)
#     print("属性特征扩展测试")
#     print("=" * 70)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"使用设备: {device}")

#     torch.manual_seed(789)

#     # 创建模拟数据
#     N_total = 30      # 总原子数
#     N_centers = 8     # 中心点数
#     K = 5             # 每个中心的邻居数
#     Ci = 10           # 属性特征维度（例如：原子类型的 one-hot 编码）

#     global_coords = torch.randn(N_total, 3, device=device, requires_grad=True)
#     global_features = torch.randn(N_total, Ci, device=device, requires_grad=True)
#     neighbor_indices = torch.randint(0, N_total, (N_centers, K), device=device)

#     print(f"\n数据规模:")
#     print(f"  总原子数 (N_total): {N_total}")
#     print(f"  中心点数 (N_centers): {N_centers}")
#     print(f"  邻居数 (K): {K}")
#     print(f"  属性特征维度 (Ci): {Ci}")

#     # 创建转换器
#     transformer = CoordinateTransformerTorch(center_method='mean').to(device)

#     # 前向传播（带属性特征）
#     print(f"\n前向传播（带属性特征）:")
#     spherical_features, centers, eigenvalues, local_features = transformer(
#         global_coords, neighbor_indices, global_features
#     )

#     print(f"  坐标输入: {global_coords.shape} → 球坐标输出: {spherical_features.shape}")
#     print(f"  属性输入: {global_features.shape} → 局部属性输出: {local_features.shape}")
#     print(f"  中心点: {centers.shape}")
#     print(f"  特征值: {eigenvalues.shape}")

#     # 验证形状
#     assert spherical_features.shape == (N_centers, K, 3), "球坐标形状错误"
#     assert local_features.shape == (N_centers, K, Ci), "局部属性形状错误"
#     assert centers.shape == (N_centers, 3), "中心点形状错误"
#     assert eigenvalues.shape == (N_centers, 3), "特征值形状错误"

#     print(f"\n✓ 输出形状验证通过")

#     # 测试属性特征的梯度传播
#     print(f"\n测试属性特征的梯度传播:")
#     loss = local_features.sum()
#     loss.backward()

#     print(f"  global_features 梯度形状: {global_features.grad.shape}")
#     print(f"  global_features 梯度范数: {global_features.grad.norm().item():.6f}")
#     print(f"  global_features 梯度非零元素: {(global_features.grad != 0).sum().item()} / {global_features.grad.numel()}")

#     # 验证梯度的正确性：只有被选中的原子应该有梯度
#     selected_indices = neighbor_indices.flatten().unique()
#     print(f"  被选中的原子索引数: {len(selected_indices)}")

#     print(f"\n✓ 属性特征梯度传播成功")

#     # 测试没有属性特征的情况
#     print(f"\n前向传播（不带属性特征）:")
#     spherical_features2, centers2, eigenvalues2, local_features2 = transformer(
#         global_coords, neighbor_indices, None
#     )

#     print(f"  球坐标输出: {spherical_features2.shape}")
#     print(f"  局部属性输出: {local_features2}")

#     assert local_features2 is None, "不提供属性时应返回 None"

#     print(f"\n✓ 可选属性特征功能正常")

#     return True


# def test_gradient_flow():
# """
# 测试梯度流动的完整性
# """
# print("\n" + "=" * 70)
# print("梯度流动测试")
# print("=" * 70)

# device = 'cpu'  # CPU 更容易调试
# torch.manual_seed(456)

# # 创建简单的测试案例
# N_total = 20
# N_centers = 5
# K = 4

# global_coords = torch.randn(N_total, 3, device=device, requires_grad=True)
# neighbor_indices = torch.randint(0, N_total, (N_centers, K), device=device)

# transformer = CoordinateTransformerTorch().to(device)

# # 前向传播
# spherical_features, centers, eigenvalues, local_features = transformer(
#     global_coords, neighbor_indices
# )

# # 对每个输出分别测试梯度
# print("\n1. 球坐标特征的梯度:")
# loss1 = spherical_features.sum()
# loss1.backward(retain_graph=True)
# grad1_norm = global_coords.grad.norm().item()
# print(f"   梯度范数: {grad1_norm:.6f}")
# global_coords.grad.zero_()

# print("\n2. 中心点的梯度:")
# loss2 = centers.sum()
# loss2.backward(retain_graph=True)
# grad2_norm = global_coords.grad.norm().item()
# print(f"   梯度范数: {grad2_norm:.6f}")
# global_coords.grad.zero_()

# print("\n3. 特征值的梯度:")
# loss3 = eigenvalues.sum()
# loss3.backward()
# grad3_norm = global_coords.grad.norm().item()
# print(f"   梯度范数: {grad3_norm:.6f}")

# print(f"\n✓ 所有输出都能传播梯度到输入")

# return True


# # def test_expand_feature_matrix():
#     """
#     测试特征矩阵扩展功能
#     """
#     print("\n" + "=" * 70)
#     print("特征矩阵扩展测试")
#     print("=" * 70)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"使用设备: {device}")

#     torch.manual_seed(999)

#     # 创建测试数据
#     N = 10      # 节点数
#     n = 5       # 每个节点的数据维度（例如邻居数）
#     Ci = 8      # 特征维度

#     data = torch.randn(N, n, device=device, requires_grad=True)
#     features = torch.randn(N, Ci, device=device, requires_grad=True)

#     print(f"\n数据规模:")
#     print(f"  数据矩阵 (data): {data.shape}")
#     print(f"  特征矩阵 (features): {features.shape}")

#     # 创建转换器
#     transformer = CoordinateTransformerTorch().to(device)

#     # 测试特征扩展
#     print(f"\n前向传播:")
#     expanded = transformer.expand_feature_matrix(data, features)

#     print(f"  输入: data {data.shape} + features {features.shape}")
#     print(f"  输出: expanded {expanded.shape}")
#     print(f"  需要梯度: {expanded.requires_grad}")

#     # 验证形状
#     expected_shape = (Ci, N, n)
#     assert expanded.shape == expected_shape, f"输出形状错误: {expanded.shape} != {expected_shape}"

#     print(f"\n✓ 输出形状验证通过: {expanded.shape}")

#     # 测试梯度传播
#     print(f"\n测试梯度传播:")
#     loss = expanded.sum()
#     loss.backward()

#     print(f"  data 梯度形状: {data.grad.shape}")
#     print(f"  data 梯度范数: {data.grad.norm().item():.6f}")
#     print(f"  features 梯度形状: {features.grad.shape}")
#     print(f"  features 梯度范数: {features.grad.norm().item():.6f}")

#     # 验证梯度健康性
#     assert data.grad is not None, "data 梯度为空！"
#     assert features.grad is not None, "features 梯度为空！"
#     assert not torch.isnan(data.grad).any(), "data 梯度包含 NaN！"
#     assert not torch.isnan(features.grad).any(), "features 梯度包含 NaN！"

#     print(f"\n✓ 梯度传播成功，所有检查通过")

#     # 测试数值验证
#     print(f"\n数值验证:")
#     # 手动计算期望结果
#     data_expanded_manual = data.unsqueeze(1)  # (N, 1, n)
#     features_expanded_manual = features.unsqueeze(2)  # (N, Ci, 1)
#     expected_manual = (data_expanded_manual * features_expanded_manual).permute(1, 0, 2)

#     diff = (expanded - expected_manual).abs().max().item()
#     print(f"  与手动计算的最大差异: {diff:.2e}")

#     assert diff < 1e-6, f"数值验证失败: {diff}"

#     print(f"\n✓ 数值验证通过")

#     return True


# if __name__ == "__main__":
# print("=" * 70)
# print("坐标转换模块 - PyTorch 可微分版本测试")
# print("=" * 70)

# # # 测试 1: 基本可微分性
# # test_differentiability()

# # # 测试 2: 梯度流动
# # test_gradient_flow()


# print("\n" + "=" * 70)
# print("🎉 所有可微分性测试通过！")
# print("=" * 70)
# print("\n使用建议:")
# print("  1. 可以作为 nn.Module 嵌入到神经网络中")
# print("  2. 支持 GPU 加速，传入 device='cuda' 的张量")
# print("  3. 支持批量处理和端到端训练")
# print("  4. 注意: PCA 在特征值重复时梯度可能不稳定")
# print("=" * 70)
