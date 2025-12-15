import numpy as np

from aggregation import AggregationLayer


def check_gradient_numerical(layer, features, weights, epsilon=1e-5):
    """
    数值梯度检查工具。
    原理: 导数定义 f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    """
    print("\n--- Running Numerical Gradient Check ---")

    # 1. 随机生成一个"假"的 Loss 梯度
    # 假设 Loss = Sum(Output), 那么 grad_output 全为 1
    # 或者随机生成 grad_output，这里我们计算 d(Sum(Y))/dX，相当于 grad_output=1
    N, Co = 10, 5  # 使用小一点的维度方便计算
    Ci, n = 3, 4

    # 重置数据
    x = np.random.randn(Ci, N, n)
    w = np.random.randn(Co, Ci, N, n)
    grad_output = np.random.randn(Co, N)

    # 2. 计算解析梯度 (Analytical Gradient)
    layer.forward(x, w)
    ana_grad_x, ana_grad_w = layer.backward(grad_output)

    # 3. 计算数值梯度 (Numerical Gradient) - 对 Features (X)
    num_grad_x = np.zeros_like(x)

    # 遍历 x 的每一个元素 (为了效率只抽查几个)
    it = np.nditer(x, flags=["multi_index"])
    print("Checking Features Gradient (sampling)...")
    count = 0
    while not it.finished and count < 10:
        idx = it.multi_index
        orig_val = x[idx]

        # x + eps
        x[idx] = orig_val + epsilon
        out_plus = layer.forward(x, w)
        loss_plus = np.sum(out_plus * grad_output)  # 模拟 Loss = Sum(Y * grad_target)

        # x - eps
        x[idx] = orig_val - epsilon
        out_minus = layer.forward(x, w)
        loss_minus = np.sum(out_minus * grad_output)

        # 恢复
        x[idx] = orig_val

        # 计算导数
        num_grad = (loss_plus - loss_minus) / (2 * epsilon)
        ana_grad = ana_grad_x[idx]

        # 比较
        diff = abs(num_grad - ana_grad)
        if diff > 1e-4:
            print(
                f"❌ Mismatch at {idx}: Num={num_grad:.6f}, Ana={ana_grad:.6f}, Diff={diff}"
            )
            return False

        it.iternext()
        count += 1

    print("✅ Gradient check for Features passed!")
    return True


def test_shapes():
    print("\n--- Testing Shapes ---")
    N, n = 100, 8
    Ci, Co = 16, 32

    feat = np.random.randn(Ci, N, n)
    weights = np.random.randn(Co, Ci, N, n)

    layer = AggregationLayer()

    # Forward
    out = layer.forward(feat, weights)
    expected_shape = (Co, N)
    if out.shape == expected_shape:
        print(f"✅ Forward Shape Correct: {out.shape}")
    else:
        print(f"❌ Forward Shape Mismatch: Got {out.shape}, Expected {expected_shape}")

    # Backward
    grad_out = np.random.randn(Co, N)
    grad_x, grad_w = layer.backward(grad_out)

    if grad_x.shape == feat.shape:
        print(f"✅ Backward Feature Shape Correct: {grad_x.shape}")
    else:
        print(f"❌ Backward Feature Shape Mismatch: {grad_x.shape}")

    if grad_w.shape == weights.shape:
        print(f"✅ Backward Weight Shape Correct: {grad_w.shape}")
    else:
        print(f"❌ Backward Weight Shape Mismatch: {grad_w.shape}")


if __name__ == "__main__":
    test_shapes()

    # 简单构造 layer 用于梯度检查
    layer = AggregationLayer()
    # 这里的输入参数在函数内部会重新生成随机数，传 None 占位即可
    check_gradient_numerical(layer, None, None)
