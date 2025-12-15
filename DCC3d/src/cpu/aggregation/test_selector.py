import numpy as np

from selector import SelectorFactory


def get_dist_matrix(coords):
    """辅助函数：计算距离矩阵 (N, N)"""
    # 简单的广播计算，用于 Ground Truth 验证
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def test_knn_accuracy():
    print("\n=== Test 1: KNN Logic Accuracy ===")
    N, n = 100, 10
    coords = np.random.rand(N, 3)

    # 1. 运行算法
    selector = SelectorFactory.get_selector({"type": "knn", "n": n})
    indices = selector.select(coords)  # (N, n)

    # 2. 计算 Ground Truth (暴力计算)
    dist_mat = get_dist_matrix(coords)
    # 对每行距离进行排序，取前 n 个的索引
    true_indices = np.argsort(dist_mat, axis=1)[:, :n]

    # 3. 验证
    # 注意: KNNSelector 可能会用 argpartition，导致内部顺序可能不同（例如第5和第6近的点顺序互换）
    # 但这 n 个点的集合必须是一样的。

    # 我们通过比较选出的点到中心的距离总和来验证
    # 选出的点的距离
    batch_indices = np.arange(N)[:, None]
    selected_dists = dist_mat[batch_indices, indices]
    true_dists = dist_mat[batch_indices, true_indices]

    # 对距离进行排序后再比较（消除顺序差异）
    selected_dists.sort(axis=1)
    true_dists.sort(axis=1)

    error = np.abs(selected_dists - true_dists).max()
    print(f"Max distance difference from Ground Truth: {error:.6f}")

    if error < 1e-5:
        print("✅ KNN selects the mathematically closest points.")
    else:
        print("❌ KNN logic error: Selected points are not the closest ones.")


def test_ball_query_radius_and_padding():
    print("\n=== Test 2: Ball Query Radius & Padding Logic ===")
    N, n = 50, 8
    radius = 0.3
    coords = np.random.rand(N, 3)

    selector = SelectorFactory.get_selector(
        {"type": "ball_query", "n": n, "radius": radius}
    )
    indices = selector.select(coords)

    dist_mat = get_dist_matrix(coords)

    # 检查 1: 半径约束
    # 获取所有被选中的点与中心的距离
    batch_indices = np.arange(N)[:, None]
    selected_dists = dist_mat[batch_indices, indices]

    # 找出最大的距离
    max_selected_dist = selected_dists.max()
    print(
        f"Radius set to: {radius}, Max distance found in selected: {max_selected_dist:.4f}"
    )

    if max_selected_dist <= radius + 1e-5:
        print("✅ Radius constraint satisfied.")
    else:
        print("❌ Radius constraint violated!")

    # 检查 2: 填充逻辑 (Padding)
    # 我们制造一个孤立点，确保它周围没有点
    coords[0] = [100, 100, 100]  # 飞得很远
    dist_mat = get_dist_matrix(coords)  # 重新计算
    indices = selector.select(coords)

    # 对于孤立点 (index 0)，除了它自己(距离0)，其他点都在半径外
    # 所以应该全部填充为 0
    point_0_indices = indices[0]
    print(f"Isolated point indices: {point_0_indices}")

    if np.all(point_0_indices == 0):
        print("✅ Padding logic correct: Isolated point refers to itself.")
    else:
        print("❌ Padding logic error.")


def test_dilated_knn_skipping():
    print("\n=== Test 3: Dilated KNN Skipping Logic ===")
    # 构造一个线性排列的点云，方便验证 "跳过" 逻辑
    # 点: (0,0,0), (1,0,0), (2,0,0), ...
    N = 20
    coords = np.zeros((N, 3))
    coords[:, 0] = np.arange(N)

    n = 3
    dilation = 2
    # 预期: 对于点 0，最近的是 0, 1, 2, 3, 4, 5...
    # Dilation=2 意味着取第 0, 2, 4 个最近的
    # 即索引应该为 [0, 2, 4]

    selector = SelectorFactory.get_selector(
        {"type": "dilated", "n": n, "dilation": dilation}
    )
    indices = selector.select(coords)

    center_idx = 0
    selected = indices[center_idx]
    # 对结果排序方便比较
    selected.sort()

    expected = np.array([0, 2, 4])
    print(f"Linear Point Cloud (0,1,2...). Dilation={dilation}, n={n}")
    print(f"Center: 0. Selected: {selected}")
    print(f"Expected: {expected}")

    if np.array_equal(selected, expected):
        print("✅ Dilated KNN correctly skips neighbors.")
    else:
        print("❌ Dilated KNN logic error.")


def test_output_shapes():
    print("\n=== Test 4: Basic Shapes & Types ===")
    N, n = 30, 5
    coords = np.random.rand(N, 3)

    configs = [
        {"type": "knn", "n": n},
        {"type": "ball_query", "n": n, "radius": 0.2},
        {"type": "dilated", "n": n, "dilation": 2},
    ]

    all_pass = True
    for cfg in configs:
        selector = SelectorFactory.get_selector(cfg)
        idx = selector.select(coords)

        # 检查形状
        if idx.shape != (N, n):
            print(f"❌ {cfg['type']} Shape mismatch: {idx.shape} != {(N, n)}")
            all_pass = False

        # 检查类型
        if not np.issubdtype(idx.dtype, np.integer):
            print(f"❌ {cfg['type']} Type mismatch: {idx.dtype} is not integer")
            all_pass = False

        # 检查范围
        if idx.min() < 0 or idx.max() >= N:
            print(f"❌ {cfg['type']} Index out of bounds: [{idx.min()}, {idx.max()}]")
            all_pass = False

    if all_pass:
        print("✅ All selectors return correct shape and type.")


if __name__ == "__main__":
    test_knn_accuracy()
    test_ball_query_radius_and_padding()
    test_dilated_knn_skipping()
    test_output_shapes()
