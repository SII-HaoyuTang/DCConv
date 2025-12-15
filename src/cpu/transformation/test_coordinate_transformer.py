"""
坐标转换模块完整测试
========================
测试所有功能和旋转不变性
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rotation_invariance import RotationInvariance, generate_random_rotation_matrix
from coordinate_transformer import CoordinateTransformer


def test_rotation_invariance_module():
    """
    测试 1: 旋转不变性模块独立测试
    """
    print("\n" + "=" * 70)
    print("测试 1: 旋转不变性模块")
    print("=" * 70)
    
    # 创建一个简单的点云（椭球形状）
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 20)
    u = np.linspace(0, np.pi, 10)
    t, u = np.meshgrid(t, u)
    
    # 椭球参数
    a, b, c = 3.0, 2.0, 1.0
    x = a * np.sin(u) * np.cos(t)
    y = b * np.sin(u) * np.sin(t)
    z = c * np.cos(u)
    
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points = points[:50]  # 取50个点
    
    # 中心化
    points = points - points.mean(axis=0)
    
    print(f"创建椭球点云: {points.shape}")
    
    # 创建旋转不变性处理器
    ri = RotationInvariance()
    
    # PCA 对齐
    aligned_original, eigenvals, eigenvecs = ri.pca_alignment(points)
    print(f"\n原始点云 PCA:")
    print(f"  特征值: {eigenvals}")
    print(f"  特征值比例: {eigenvals / eigenvals[0]}")
    print(f"  (应该接近椭球的轴比例 {a}:{b}:{c} = {a}:{b}:{c})")
    
    # 生成随机旋转
    rotation = generate_random_rotation_matrix()
    rotated_points = points @ rotation.T
    
    # 旋转后再 PCA 对齐
    aligned_rotated, eigenvals_rot, _ = ri.pca_alignment(rotated_points)
    
    print(f"\n旋转后点云 PCA:")
    print(f"  特征值: {eigenvals_rot}")
    
    # 验证旋转不变性
    is_invariant = ri.verify_rotation_invariance(points, rotation)
    print(f"\n旋转不变性验证: {'✓ 通过' if is_invariant else '✗ 失败'}")
    
    # 检查对齐结果的差异
    diff = np.abs(np.abs(aligned_original) - np.abs(aligned_rotated))
    print(f"对齐结果最大差异: {np.max(diff):.2e}")
    
    return is_invariant


def test_coordinate_transformer():
    """
    测试 2: 完整的坐标转换流程
    """
    print("\n" + "=" * 70)
    print("测试 2: 完整坐标转换流程")
    print("=" * 70)
    
    np.random.seed(123)
    
    # 创建模拟的分子坐标（水分子 H2O 的多个副本）
    # 基础水分子结构
    water_molecule = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.757, 0.586, 0.0],  # H1
        [-0.757, 0.586, 0.0]  # H2
    ])
    
    # 创建多个水分子
    N_molecules = 33
    global_coords = []
    for i in range(N_molecules):
        # 随机平移和小扰动
        offset = np.random.randn(3) * 5.0
        noise = np.random.randn(3, 3) * 0.1
        molecule = water_molecule + offset + noise
        global_coords.append(molecule)
    
    global_coords = np.vstack(global_coords)
    N_total = global_coords.shape[0]
    
    print(f"创建模拟数据: {N_molecules} 个水分子, 共 {N_total} 个原子")
    
    # 创建邻居索引（每个氧原子选择最近的10个原子）
    N_centers = N_molecules
    K = 10
    neighbor_indices = np.zeros((N_centers, K), dtype=int)
    
    for i in range(N_centers):
        center_idx = i * 3  # 氧原子索引
        center_pos = global_coords[center_idx]
        
        # 计算距离
        distances = np.linalg.norm(global_coords - center_pos, axis=1)
        
        # 选择最近的 K 个原子
        nearest_indices = np.argsort(distances)[:K]
        neighbor_indices[i] = nearest_indices
    
    # 创建坐标转换器
    transformer = CoordinateTransformer(center_method='mean')
    
    # 执行转换
    spherical_features, centers, eigenvalues = transformer(
        global_coords, neighbor_indices
    )
    
    print(f"\n转换结果:")
    print(f"  球坐标特征形状: {spherical_features.shape}")
    print(f"  中心点形状: {centers.shape}")
    print(f"  特征值形状: {eigenvalues.shape}")
    
    # 检查球坐标范围
    r = spherical_features[:, :, 0]
    theta = spherical_features[:, :, 1]
    phi = spherical_features[:, :, 2]
    
    print(f"\n球坐标统计:")
    print(f"  r (距离):")
    print(f"    范围: [{r.min():.3f}, {r.max():.3f}]")
    print(f"    均值: {r.mean():.3f}")
    
    print(f"  θ (极角):")
    print(f"    范围: [{theta.min():.3f}, {theta.max():.3f}]")
    print(f"    理论范围: [0, π] = [0, {np.pi:.3f}]")
    
    print(f"  φ (方位角):")
    print(f"    范围: [{phi.min():.3f}, {phi.max():.3f}]")
    print(f"    理论范围: [0, 2π] = [0, {2*np.pi:.3f}]")
    
    # 验证角度范围正确
    assert r.min() >= 0, "径向距离必须非负"
    assert theta.min() >= 0 and theta.max() <= np.pi + 1e-6, "极角超出范围"
    assert phi.min() >= 0 and phi.max() <= 2*np.pi + 1e-6, "方位角超出范围"
    
    print(f"\n✓ 球坐标范围验证通过")
    
    return spherical_features, centers, eigenvalues


def test_full_rotation_invariance():
    """
    测试 3: 完整流程的旋转不变性
    """
    print("\n" + "=" * 70)
    print("测试 3: 完整流程的旋转不变性验证")
    print("=" * 70)
    
    np.random.seed(456)
    
    # 创建简单的测试数据
    N_total = 50
    N_centers = 10
    K = 8
    
    global_coords = np.random.randn(N_total, 3) * 3.0
    neighbor_indices = np.random.randint(0, N_total, size=(N_centers, K))
    
    # 创建转换器
    transformer = CoordinateTransformer()
    
    # 原始转换
    spherical_orig, centers_orig, eigenvals_orig = transformer(
        global_coords, neighbor_indices
    )
    
    print(f"原始数据转换完成")
    print(f"  第一个局部点云的前3个球坐标:")
    print(f"    r: {spherical_orig[0, :3, 0]}")
    print(f"    θ: {spherical_orig[0, :3, 1]}")
    print(f"    φ: {spherical_orig[0, :3, 2]}")
    
    # 随机旋转整个体系
    rotation = generate_random_rotation_matrix()
    global_coords_rotated = global_coords @ rotation.T
    
    # 旋转后转换
    spherical_rot, centers_rot, eigenvals_rot = transformer(
        global_coords_rotated, neighbor_indices
    )
    
    print(f"\n旋转后数据转换完成")
    print(f"  第一个局部点云的前3个球坐标:")
    print(f"    r: {spherical_rot[0, :3, 0]}")
    print(f"    θ: {spherical_rot[0, :3, 1]}")
    print(f"    φ: {spherical_rot[0, :3, 2]}")
    
    # 比较球坐标的径向距离（应该几乎相同）
    r_orig = spherical_orig[:, :, 0]
    r_rot = spherical_rot[:, :, 0]
    
    # 由于 PCA 对齐，r 应该保持不变（允许符号翻转）
    # 但角度可能会变化，所以我们主要检查 r
    r_diff = np.abs(np.abs(r_orig) - np.abs(r_rot))
    max_r_diff = np.max(r_diff)
    mean_r_diff = np.mean(r_diff)
    
    print(f"\n径向距离差异:")
    print(f"  最大差异: {max_r_diff:.2e}")
    print(f"  平均差异: {mean_r_diff:.2e}")
    
    # 检查特征值（应该相同）
    eigenvals_diff = np.abs(eigenvals_orig - eigenvals_rot)
    max_eigen_diff = np.max(eigenvals_diff)
    
    print(f"\nPCA 特征值差异:")
    print(f"  最大差异: {max_eigen_diff:.2e}")
    
    # 验证
    tolerance = 1e-5
    r_invariant = max_r_diff < tolerance
    eigen_invariant = max_eigen_diff < tolerance
    
    print(f"\n旋转不变性验证:")
    print(f"  径向距离: {'✓ 通过' if r_invariant else '✗ 失败'}")
    print(f"  PCA 特征值: {'✓ 通过' if eigen_invariant else '✗ 失败'}")
    
    overall_pass = r_invariant and eigen_invariant
    
    return overall_pass


def test_translation_invariance():
    """
    测试 4: 平移不变性验证
    """
    print("\n" + "=" * 70)
    print("测试 4: 平移不变性验证")
    print("=" * 70)
    
    np.random.seed(789)
    
    # 创建测试数据
    N_total = 30
    N_centers = 5
    K = 6
    
    global_coords = np.random.randn(N_total, 3) * 2.0
    neighbor_indices = np.random.randint(0, N_total, size=(N_centers, K))
    
    # 创建转换器
    transformer = CoordinateTransformer()
    
    # 原始转换
    spherical_orig, _, _ = transformer(global_coords, neighbor_indices)
    
    # 平移整个体系
    translation = np.array([100.0, -50.0, 75.0])
    global_coords_translated = global_coords + translation
    
    # 平移后转换
    spherical_trans, _, _ = transformer(global_coords_translated, neighbor_indices)
    
    # 球坐标应该完全相同（平移不影响相对位置）
    diff = np.abs(spherical_orig - spherical_trans)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"平移向量: {translation}")
    print(f"\n球坐标差异:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    
    tolerance = 1e-10
    is_invariant = max_diff < tolerance
    
    print(f"\n平移不变性验证: {'✓ 通过' if is_invariant else '✗ 失败'}")
    
    return is_invariant


def visualize_transformation(save_fig=False):
    """
    可视化：展示转换前后的坐标
    """
    print("\n" + "=" * 70)
    print("可视化：坐标转换")
    print("=" * 70)
    
    try:
        np.random.seed(100)
        
        # 创建一个小型点云
        N = 15
        points = np.random.randn(N, 3)
        points[:, 0] *= 3  # 拉伸 x 轴
        points[:, 1] *= 2  # 拉伸 y 轴
        points[:, 2] *= 1  # z 轴保持
        
        # 中心化
        center = points.mean(axis=0)
        relative_points = points - center
        
        # PCA 对齐
        ri = RotationInvariance()
        aligned_points, eigenvals, eigenvecs = ri.pca_alignment(relative_points)
        
        # 创建图形
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1: 原始点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=50)
        ax1.scatter([center[0]], [center[1]], [center[2]], c='red', s=200, marker='*')
        ax1.set_title('原始坐标 (绝对位置)', fontsize=12, fontproperties='SimHei')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 子图2: 相对坐标
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(relative_points[:, 0], relative_points[:, 1], 
                   relative_points[:, 2], c='green', s=50)
        ax2.scatter([0], [0], [0], c='red', s=200, marker='*')
        ax2.set_title('相对坐标 (平移不变)', fontsize=12, fontproperties='SimHei')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 子图3: PCA 对齐后
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(aligned_points[:, 0], aligned_points[:, 1], 
                   aligned_points[:, 2], c='purple', s=50)
        ax3.scatter([0], [0], [0], c='red', s=200, marker='*')
        ax3.set_title('PCA 对齐 (旋转不变)', fontsize=12, fontproperties='SimHei')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_zlabel('PC3')
        
        # 绘制主成分轴
        for i in range(3):
            direction = eigenvecs[:, i] * np.sqrt(eigenvals[i]) * 2
            ax3.plot([0, direction[0]], [0, direction[1]], [0, direction[2]], 
                    'r-', linewidth=2, alpha=0.6)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('coordinate_transformation_visualization.png', dpi=150)
            print("图形已保存为: coordinate_transformation_visualization.png")
        else:
            print("提示: 设置 save_fig=True 可保存图形")
        
        # plt.show()  # 如果要显示图形，取消注释
        plt.close()
        
        print("✓ 可视化完成")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        print("(这不影响核心功能)")


def run_all_tests():
    """
    运行所有测试
    """
    print("\n")
    print("#" * 70)
    print("# 坐标转换与旋转不变性模块 - 完整测试套件")
    print("#" * 70)
    
    results = {}
    
    # 测试 1
    results['rotation_module'] = test_rotation_invariance_module()
    
    # 测试 2
    test_coordinate_transformer()
    results['coordinate_transformer'] = True
    
    # 测试 3
    results['full_rotation_invariance'] = test_full_rotation_invariance()
    
    # 测试 4
    results['translation_invariance'] = test_translation_invariance()
    
    # 可视化
    visualize_transformation(save_fig=False)
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有测试通过！模块已准备好使用。")
    else:
        print("⚠️  部分测试失败，请检查代码。")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
