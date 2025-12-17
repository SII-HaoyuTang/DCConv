import pytest
import numpy as np
from DCC3d.src.cpu.selector.selector_numpy import (
    calc_pairwise_distance, 
    BallQuerySelector, 
    DilatedKNNSelector, 
    SelectorFactory, 
    SelectorType,
    KNNSelector
)

# ==========================================
# 1. Fixtures (数据准备)
# ==========================================

@pytest.fixture
def simple_line_coords():
    """生成一条直线上的点: (0,0,0), (1,0,0), ... (5,0,0)"""
    return np.array([[i, 0, 0] for i in range(6)], dtype=np.float32)

@pytest.fixture
def dense_cluster():
    """生成一个密集的点簇，用于测试截断"""
    return np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ], dtype=np.float32)

@pytest.fixture
def large_coords_small_diff():
    """生成大坐标小距离的数据，测试数值稳定性"""
    offset = 1e5
    epsilon = 0.1
    return np.array([
        [offset, 0, 0],
        [offset + epsilon, 0, 0]
    ], dtype=np.float64), epsilon

# ==========================================
# 2. 核心算法测试 (Distance)
# ==========================================

def test_distance_numerical_stability(large_coords_small_diff):
    """测试在大坐标下计算微小距离的数值稳定性"""
    coords, expected_dist = large_coords_small_diff
    dist_mat = calc_pairwise_distance(coords)
    
    calculated = dist_mat[0, 1]

    assert np.isclose(calculated, expected_dist, atol=1e-4), \
        f"数值不稳定: 预期 {expected_dist}, 实际 {calculated}"

    assert np.all(np.diag(dist_mat) == 0.0)

def test_distance_matrix_shape(simple_line_coords):
    """测试距离矩阵形状"""
    N = simple_line_coords.shape[0]
    dist = calc_pairwise_distance(simple_line_coords)
    assert dist.shape == (N, N)

# ==========================================
# 3. Ball Query Selector 测试
# ==========================================

class TestBallQuery:
    
    @pytest.mark.parametrize("n_sample, radius, expected_indices", [
        (3, 1.5, [0, 1, 0]),
        (3, 0.5, [0, 0, 0]),
    ])
    def test_padding_logic(self, simple_line_coords, n_sample, radius, expected_indices):
        selector = BallQuerySelector(n_sample=n_sample, radius=radius)
        indices = selector(simple_line_coords)
        
        # 检查第一个点 (0,0,0) 的邻居索引
        assert np.array_equal(indices[0], expected_indices), \
            f"填充逻辑错误. 预期 {expected_indices}, 实际 {indices[0]}"

    def test_truncation_priority(self, dense_cluster):
        """测试截断时是否保留了最近的点"""
        selector = BallQuerySelector(n_sample=3, radius=1.0)
        indices = selector(dense_cluster)
        neighbors = indices[0]
        expected_subset = {0, 1, 2} # 集合比较，忽略顺序(虽然代码通常有序)
        assert set(neighbors) == expected_subset, \
            f"截断未保留最近点. 预期包含 {expected_subset}, 实际 {neighbors}"

# ==========================================
# 4. Dilated KNN Selector 测试
# ==========================================

class TestDilatedKNN:
    
    def test_dilation_stride(self, simple_line_coords):
        """测试膨胀步长是否正确"""
        selector = DilatedKNNSelector(n_sample=2, dilation=2)
        indices = selector(simple_line_coords)

        assert np.array_equal(indices[0], [0, 2]), \
            f"膨胀采样错误. 预期 [0, 2], 实际 {indices[0]}"

    def test_insufficient_points_error(self):
        """测试点数不足时是否抛出异常"""
        coords = np.zeros((3, 3))
        selector = DilatedKNNSelector(n_sample=2, dilation=2)
        
        with pytest.raises(ValueError, match="exceeds points"):
            selector(coords)
            
    def test_fallback_behavior(self):
        """
        测试回退逻辑 (虽然当前代码有 ValueError 守卫, 
        但如果 search_range <= N 但切片后不足 n_sample, 应回退)
        """
        pass

def test_overlapping_points():
    """
    边缘场景：所有点坐标完全相同（黑洞）。
    测试目的：验证排序稳定性，以及除零/NaN风险。
    """
    coords = np.ones((5, 3), dtype=np.float32)

    selector = KNNSelector(n_sample=3)
    indices = selector(coords)
    
    assert indices.shape == (5, 3)
    assert np.all(indices >= 0) and np.all(indices < 5)

    dist = calc_pairwise_distance(coords)
    assert np.allclose(dist, 0.0)

def test_1d_point_cloud():
    """
    边缘场景：输入是 (N, 1) 的形状，而不是标准的 (N, 3)。
    测试目的：验证代码是否硬编码了维度 3。
    """
    coords = np.array([[i] for i in range(10)], dtype=np.float32)
    selector = KNNSelector(n_sample=2)
    indices = selector(coords)

    idx_5 = indices[5]
    assert 5 in idx_5
    assert (4 in idx_5) or (6 in idx_5)

# ==========================================
# 2. 脏数据注入测试 (Dirty Data)
# ==========================================

def test_nan_input_handling():
    """
    边缘场景：输入坐标包含 NaN。
    测试目的：确保计算距离矩阵时不会抛出 Python 异常（虽然结果可能是垃圾）。
    通常 NumPy 会传播 NaN。
    """
    coords = np.array([
        [0, 0, 0],
        [np.nan, np.nan, np.nan], # 坏点
        [1, 1, 1]
    ])
    
    selector = KNNSelector(n_sample=2)
    indices = selector(coords)

    assert indices.shape == (3, 2)

    dist = calc_pairwise_distance(coords)
    assert np.isnan(dist[1]).all()

def test_inf_input_handling():
    """
    边缘场景：输入坐标包含 Infinity。
    """
    coords = np.array([
        [0, 0, 0],
        [np.inf, 0, 0], # 无穷远点
        [1, 0, 0]
    ])
    
    selector = KNNSelector(n_sample=2)
    indices = selector(coords)
    
    assert 1 not in indices[0], "KNN 不应选中无穷远的点"

# ==========================================
# 3. 逻辑溢出与边界测试 (Boundary Conditions)
# ==========================================

def test_knn_k_greater_than_N():
    """
    边缘场景：请求的邻居数 k 大于总点数 N。
    预期行为：np.argpartition 会抛出 ValueError。
    注意：这是为了确认代码确实会在不支持的操作上报错（Fail Fast），而不是静默产生错误结果。
    """
    coords = np.zeros((3, 3)) # N=3
    k = 5
    selector = KNNSelector(n_sample=k)
    
    with pytest.raises(ValueError, match="bounds"):
        selector(coords)

def test_ball_query_radius_zero():
    """
    边缘场景：半径为 0。
    测试目的：根据逻辑 `dist < radius`，0 < 0 为 False。
    因此没有点满足条件。代码应触发 Padding 逻辑（填充自身）。
    """
    coords = np.array([[0,0,0], [1,0,0]])
    selector = BallQuerySelector(n_sample=2, radius=0.0)
    indices = selector(coords)
    
    assert np.all(indices[0] == 0)
    assert np.all(indices[1] == 1)

def test_ball_query_radius_negative():
    """
    边缘场景：负半径。
    预期行为：同半径为0，甚至更严格，只能选自身。
    """
    coords = np.array([[0,0,0], [1,0,0]])
    selector = BallQuerySelector(n_sample=2, radius=-1.0)
    indices = selector(coords)
    assert np.all(indices[0] == 0)

# ==========================================
# 4. 数学/算法特异性测试
# ==========================================

def test_dilated_knn_exact_boundary():
    """
    边缘场景：N 刚好等于 search_range。
    Dilation=2, n_sample=2 -> range=4.
    Points N=4.
    """
    coords = np.zeros((4, 3))
    selector = DilatedKNNSelector(n_sample=2, dilation=2)

    try:
        indices = selector(coords)
        assert indices.shape == (4, 2)
    except ValueError:
        pytest.fail("N == search_range 不应抛出异常")
    
# ==========================================
# 5. Factory 测试
# ==========================================

def test_factory_creation():
    config = {"type": "ball_query", "n": 16, "radius": 0.5}
    selector = SelectorFactory.get_selector(config)
    assert isinstance(selector, BallQuerySelector)
    assert selector.radius == 0.5
    assert selector.n_sample == 16

def test_factory_invalid_type():
    with pytest.raises(ValueError, match="Invalid selector type"):
        SelectorFactory.get_selector({"type": "magic_selector", "n": 10})