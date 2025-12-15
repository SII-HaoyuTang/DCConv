"""
坐标转换与旋转不变性模块
======================================

这个包提供了用于深度学习卷积神经网络的坐标转换和旋转不变性处理功能。
所有操作都是基于 PyTorch 实现的，完全可微分，支持端到端训练。

主要模块：
---------
1. CoordinateTransformerTorch: 坐标转换主类
   - 局部格点坐标提取
   - 中心点计算
   - 相对坐标计算（平移不变性）
   - PCA 旋转对齐（旋转不变性）
   - 笛卡尔坐标转球极坐标

2. RotationInvarianceTorch: 旋转不变性处理类
   - PCA 对齐实现旋转不变性
   - 协方差矩阵计算
   - 特征值分解（可微分）

工具函数：
----------
- generate_random_rotation_matrix_torch: 生成随机旋转矩阵（用于测试）

使用示例：
----------
```python
from transformation import CoordinateTransformerTorch

# 创建坐标转换器
transformer = CoordinateTransformerTorch(
    center_method='mean',
    use_pca=True,
    pca_stabilize=True
)

# 转换坐标
spherical_coords, spherical_features = transformer(
    global_coords=coords,
    neighbor_indices=indices,
    global_features=features
)
```

注意事项：
----------
- 所有操作都支持自动微分和反向传播
- 支持 GPU 加速（自动识别输入张量的设备）
- median 中心计算方法不可微分，训练时会自动切换为 mean
"""

from .coordinate_transformer import CoordinateTransformerTorch
from .rotation_invariance import (
    RotationInvarianceTorch,
    generate_random_rotation_matrix_torch
)

__all__ = [
    # 主要类
    'CoordinateTransformerTorch',
    'RotationInvarianceTorch',
    
    # 工具函数
    'generate_random_rotation_matrix_torch',
]

__version__ = '1.0.0'
__author__ = 'DCConv Team'
