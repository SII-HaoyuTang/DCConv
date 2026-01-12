import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class EnhancedInitialization:
    """增强的神经网络参数初始化类"""

    @staticmethod
    def xavier_normal_init(layer, gain=1.0):
        """Xavier正态分布初始化（适合tanh/sigmoid激活）"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    @staticmethod
    def xavier_uniform_init(layer, gain=1.0):
        """Xavier均匀分布初始化"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    @staticmethod
    def kaiming_normal_init(layer, nonlinearity="relu"):
        """Kaiming/He正态分布初始化（适合ReLU族激活）"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity=nonlinearity)
            if layer.bias is not None:
                # 对于ReLU，偏置初始化为小的正值可以避免"死神经元"
                if nonlinearity == "relu":
                    init.constant_(layer.bias, 0.01)
                else:
                    init.zeros_(layer.bias)

    @staticmethod
    def kaiming_uniform_init(layer, nonlinearity="relu"):
        """Kaiming均匀分布初始化"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(
                layer.weight, mode="fan_in", nonlinearity=nonlinearity
            )
            if layer.bias is not None:
                init.zeros_(layer.bias)

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        """正交初始化（保持梯度范数）"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    @staticmethod
    def sparse_init(layer, sparsity=0.1, std=0.01):
        """稀疏初始化"""
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # 先使用正态分布初始化
            init.normal_(layer.weight, mean=0, std=std)

            # 设置一部分权重为0
            with torch.no_grad():
                mask = torch.rand(layer.weight.shape) > sparsity
                layer.weight.mul_(mask.float())

            if layer.bias is not None:
                init.zeros_(layer.bias)

    @staticmethod
    def spectral_normalization(layer, power_iterations=1):
        """谱归一化（稳定训练）"""
        # 这通常作为包装器使用
        return nn.utils.spectral_norm(layer, n_power_iterations=power_iterations)

    @staticmethod
    def layer_specific_init(model, config):
        """
        为不同层应用不同的初始化策略

        参数:
            model: 神经网络模型
            config: 初始化配置字典
                {
                    'conv': 'kaiming_normal',
                    'linear': 'xavier_normal',
                    'bn': {'weight': 'ones', 'bias': 'zeros'},
                    'lstm': 'orthogonal',
                }
        """

        def init_weights(m):
            classname = m.__class__.__name__

            if "Conv" in classname:
                init_method = config.get("conv", "kaiming_normal")
                if init_method == "kaiming_normal":
                    EnhancedInitialization.kaiming_normal_init(m)
                elif init_method == "xavier_normal":
                    EnhancedInitialization.xavier_normal_init(m)
                elif init_method == "orthogonal":
                    EnhancedInitialization.orthogonal_init(m)

            elif "Linear" in classname:
                init_method = config.get("linear", "xavier_normal")
                if init_method == "kaiming_normal":
                    EnhancedInitialization.kaiming_normal_init(
                        m, nonlinearity="leaky_relu"
                    )
                elif init_method == "xavier_normal":
                    EnhancedInitialization.xavier_normal_init(m)
                elif init_method == "orthogonal":
                    EnhancedInitialization.orthogonal_init(m, gain=np.sqrt(2))

            elif "BatchNorm" in classname:
                # BatchNorm通常权重初始化为1，偏置为0
                init_method = config.get("bn", {})
                weight_init = init_method.get("weight", "ones")
                bias_init = init_method.get("bias", "zeros")

                if weight_init == "ones":
                    init.ones_(m.weight)
                elif weight_init == "uniform":
                    init.uniform_(m.weight, 0.9, 1.1)

                if bias_init == "zeros":
                    init.zeros_(m.bias)

            elif "LSTM" in classname or "GRU" in classname:
                # RNN的特殊初始化
                for name, param in m.named_parameters():
                    if "weight" in name:
                        EnhancedInitialization.orthogonal_init(nn.Parameter(param))
                    elif "bias" in name:
                        # LSTM的偏置通常初始化为特定值
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)  # 遗忘门偏置
                        init.zeros_(param.data[: n // 4])  # 输入门偏置
                        init.zeros_(param.data[n // 2 : 3 * n // 4])  # 细胞门偏置
                        init.zeros_(param.data[3 * n // 4 :])  # 输出门偏置

        model.apply(init_weights)
        return model
