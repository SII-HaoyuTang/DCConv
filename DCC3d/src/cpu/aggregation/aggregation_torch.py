import torch


class AggregationLayer:
    r"""
    聚合层 (Aggregation Layer)

    对应流程图左下角的操作：
    1. 接收输入特征 (Features) 和动态权重 (Weights)。
    2. 执行元素级相乘 (Element-wise Multiplication)。
    3. 沿着输入通道 ($C_i$) 和邻居维度 ($n$) 进行求和压缩。

    Mathematical Formulation:
        Forward:
        $Y_{o, u} = \sum_{k=1}^{n} \sum_{c=1}^{C_i} W_{o, c, u, k} \cdot X_{c, u, k}$

        其中:
        - $u$: 点的索引 (1...N)
        - $k$: 邻居索引 (1...n)
        - $c$: 输入通道 (1...Ci)
        - $o$: 输出通道 (1...Co)
    """

    def __init__(self):
        self.cache: tuple[torch.Tensor, torch.Tensor] = None

    def forward(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (Forward Pass).

        Args:
            features (torch.Tensor): 邻域特征矩阵。
                Shape: (Ci, N, n)
            weights (torch.Tensor): 动态生成的卷积核权重。
                Shape: (Co, Ci, N, n)

        Returns:
            output (torch.Tensor): 聚合后的特征。
                Shape: (Co, N)
        """
        if features.shape[0] != weights.shape[1]:
            raise ValueError(
                f"Channel mismatch! Features has Ci={features.shape[0]}, "
                f"but Weights expects Ci={weights.shape[1]}"
            )
        if features.shape[1:] != weights.shape[2:]:
            raise ValueError(
                f"Spatial dimension mismatch! "
                f"Features: {features.shape[1:]} vs Weights: {weights.shape[2:]}"
            )

        self.cache = (features, weights)

        # Einstein Summation
        # c: Ci (Input Channels)
        # u: N  (Number of Points)
        # k: n  (Number of Neighbors)
        # o: Co (Output Channels)
        # sum_over_k( sum_over_c( feat[c,u,k] * weight[o,c,u,k] ) )
        output = torch.einsum("cuk, ocuk -> ou", features, weights)

        return output

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        反向传播 (Backward Pass).

        根据链式法则计算对输入 Features 和 Weights 的梯度。

        Args:
            grad_output (torch.Tensor): 来自下一层的梯度 $\partial L / \partial Y$。
                Shape: (Co, N)

        Returns:
            grad_features (torch.Tensor): 对输入特征的梯度。Shape: (Ci, N, n)
            grad_weights (torch.Tensor): 对权重的梯度。Shape: (Co, Ci, N, n)
        """
        if self.cache is None:
            raise RuntimeError("Forward must be called before backward.")

        features, weights = self.cache

        # -----------------------------------------------------------
        # 推导逻辑:
        # 1. grad_features (对 X 求导):
        #    dL/dX_{c,u,k} = sum_o( dL/dY_{o,u} * dY_{o,u}/dX_{c,u,k} )
        #    由 Forward 公式可知: dY_{o,u}/dX_{c,u,k} = W_{o,c,u,k}
        #    所以: grad_X = sum_o( grad_out[o,u] * W[o,c,u,k] )
        #    Einsum: 'ou, ocuk -> cuk'
        # -----------------------------------------------------------
        grad_features = torch.einsum("ou, ocuk -> cuk", grad_output, weights)

        # -----------------------------------------------------------
        # 2. grad_weights (对 W 求导):
        #    dL/dW_{o,c,u,k} = dL/dY_{o,u} * dY_{o,u}/dW_{o,c,u,k}
        #    由 Forward 公式可知: dY_{o,u}/dW_{o,c,u,k} = X_{c,u,k}
        #    所以: grad_W = grad_out[o,u] * X[c,u,k]
        #    注意这里没有求和，只是广播相乘 (Outer Product on specific dims)
        #    Einsum: 'ou, cuk -> ocuk'
        # -----------------------------------------------------------
        grad_weights = torch.einsum("ou, cuk -> ocuk", grad_output, features)

        return grad_features, grad_weights