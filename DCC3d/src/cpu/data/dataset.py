from typing import Dict, List, Optional
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def fix_decimal_point(value):
    """
    修复基数部分的小数点问题
    """
    # 使用正则表达式找到科学计数法格式
    pattern = r"([+-]?\d*\.?\d*)[eE]([+-]?\d+)"
    match = re.match(pattern, value)

    if match:
        num = match.group(1)
        exp = match.group(2)

        # 修复各种小数点格式问题
        if num == "" or num in ["+", "-"]:
            num = num + "1"
        elif num.endswith("."):
            num = num + "0"
        elif num.startswith(".") and not num.startswith("-."):
            num = "0" + num
        elif num.startswith("-."):
            num = "-0." + num[2:]

        return f"{num}e{exp}"

    return value


def clean_scientific_notation(value):
    """
    更健壮的清理函数，处理各种边缘情况
    """
    if pd.isna(value):
        return value

    str_value = str(value).strip()

    # 如果已经是数字，直接返回
    try:
        float(str_value)
        return str_value
    except ValueError:
        pass

    # 常见的非标准科学计数法模式
    patterns = [
        # 模式1: 数字E/e^数字（如 -6.E^-6）
        (r"([+-]?\d*\.?\d*)[Ee]\^([+-]?\d+)", r"\1e\2"),
        # 模式2: 数字E/e^+数字（如 1.23E^+4）
        (r"([+-]?\d*\.?\d*)[Ee]\^\+(\d+)", r"\1e+\2"),
        # 模式3: 数字E/e^-数字（如 5.67e^-3）
        (r"([+-]?\d*\.?\d*)[Ee]\^-(\d+)", r"\1e-\2"),
        # 模式4: 数字E/e数字（没有^，但可能有问题）
        (r"([+-]?\d*\.?\d*)[Ee](\d+)", r"\1e\2"),
    ]

    original = str_value
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, original)
        if cleaned != original:
            # 修复基数部分的小数点问题
            cleaned = fix_decimal_point(cleaned)
            return cleaned

    # 如果正则表达式没有匹配，尝试简单替换
    if "^" in str_value:
        cleaned = str_value.replace("^", "")
        cleaned = fix_decimal_point(cleaned)
        return cleaned

    return str_value


def clean_dataframe_scientific_notation(df):
    """
    清理DataFrame中所有列的非标准科学计数法
    """
    cleaned_df = df.copy()

    for col in cleaned_df.columns:
        # 只处理字符串类型的列
        if cleaned_df[col].dtype == object:
            cleaned_df[col] = cleaned_df[col].apply(clean_scientific_notation)

    return cleaned_df


def safe_convert_to_numeric(series):
    """
    安全地将序列转换为数值类型
    """
    # 先清理科学计数法
    cleaned_series = series.apply(clean_scientific_notation)

    # 转换为数值，无法转换的设为NaN
    numeric_series = pd.to_numeric(cleaned_series, errors="coerce")

    return numeric_series


class PointCloudQM9Dataset(Dataset):
    """
    点云QM9数据集类，用于处理和加载分子点云数据。

    该类继承自`torch.utils.data.Dataset`，提供了从CSV文件读取点云数据的功能，并支持数据变换。
    它主要用于处理包含分子结构信息的数据集，例如QM9数据集。通过提供分子ID、坐标、节点特征等信息，
    可以方便地进行分子性质预测任务。

    参数:
        points_csv: str
            点云CSV文件路径。
        indices_csv: str
            索引CSV文件路径。
        transform: Optional[callable]
            数据变换函数，默认为None。
        target_column: str
            目标列名，默认为"internal_energy"。
        node_features: List[str]
            节点特征列名列表，默认为None，自动推断。
        device: str
            设备，默认为"cpu"。

    方法:
        __len__
            返回数据集大小。
        __getitem__
            获取单个样本。
        get_molecule_by_id
            通过分子ID获取分子数据。
        get_molecule_smiles
            获取分子的SMILES表示（如果有）。
    """

    def __init__(
        self,
        points_csv: str,
        indices_csv: str,
        transform: Optional[callable] = None,
        target_column: str = "energy",
        node_features: List[str] = None,
        clean_scientific_notation: bool = True,
        device: str = "cpu",
    ):
        """
        Initializes a dataset for molecular data, loading points and indices from CSV files.

        Summary:
        This class is designed to load and process molecular data for machine learning tasks. It initializes with paths to
        CSV files containing point cloud data and indices, allowing for transformations and target column specification.
        The dataset supports automatic detection of node features if not provided explicitly.

        Parameters:
        - points_csv (str): Path to the CSV file containing point cloud data.
        - indices_csv (str): Path to the CSV file containing indices information.
        - transform (Optional[callable]): An optional transformation to be applied on the loaded data.
        - target_column (str): The name of the column in the dataset that is used as the target or label.
        - node_features (List[str]): A list of column names in the points CSV to be treated as node features. If not provided,
          it will automatically infer the node features excluding certain columns.
        - device (str): The device to which the data should be moved after loading. Default is "cpu".

        Attributes:
        - points_df (DataFrame): DataFrame containing the point cloud data.
        - indices_df (DataFrame): DataFrame containing the indices data, indexed by molecule_id for quick access.
        - molecule_ids (List[int]): List of unique molecule IDs present in the indices data.
        - num_molecules (int): Number of unique molecules in the dataset.
        - node_features (List[str]): List of feature column names used as node attributes.
        - device (str): Device on which the data is to be processed.
        """
        super().__init__()

        self.points_csv = points_csv
        self.indices_csv = indices_csv
        self.transform = transform
        self.target_column = target_column
        self.device = device
        self.clean_scientific_notation = clean_scientific_notation

        # 读取数据
        print("正在读取点云数据...")
        self.points_df = pd.read_csv(points_csv)
        print("正在读取索引数据...")
        self.indices_df = pd.read_csv(indices_csv)

        # 如果需要，清理非标准科学计数法
        if clean_scientific_notation:
            print("正在清理非标准科学计数法...")
            self.points_df = clean_dataframe_scientific_notation(self.points_df)
            self.indices_df = clean_dataframe_scientific_notation(self.indices_df)

        # 确保目标列是数值类型
        if target_column in self.indices_df.columns:
            self.indices_df[target_column] = safe_convert_to_numeric(
                self.indices_df[target_column]
            )
        else:
            raise ValueError(
                f"目标列 '{target_column}' 不存在于索引文件中。可用列: {self.indices_df.columns.tolist()}"
            )

        # 设置索引
        self.indices_df.set_index("molecule_id", inplace=True)

        # 获取所有唯一的分子ID
        self.molecule_ids = sorted(self.indices_df.index.unique())
        self.num_molecules = len(self.molecule_ids)

        # 如果没有指定节点特征，自动推断
        if node_features is None:
            exclude_cols = ["molecule_id", "atom_index", "x", "y", "z"]
            self.node_features = [
                col for col in self.points_df.columns if col not in exclude_cols
            ]
        else:
            self.node_features = node_features

        print(f"数据集包含 {self.num_molecules} 个分子")
        print(f"节点特征: {self.node_features}")

        # 统计信息
        self._compute_statistics()

    def _compute_statistics(self):
        """计算数据集的统计信息"""
        # 分子大小统计
        molecule_sizes = []
        for mol_id in self.molecule_ids:
            num_atoms = self.indices_df.loc[mol_id, "num_atoms"]
            molecule_sizes.append(num_atoms)

        self.max_atoms = max(molecule_sizes)
        self.min_atoms = min(molecule_sizes)
        self.avg_atoms = np.mean(molecule_sizes)

        print(
            f"分子大小统计: 最大={self.max_atoms}, 最小={self.min_atoms}, 平均={self.avg_atoms:.2f}"
        )

        # 特征统计
        self.feature_stats = {}
        for feature in self.node_features:
            if feature in self.points_df.columns:
                feature_series = safe_convert_to_numeric(self.points_df[feature])
                self.feature_stats[feature] = {
                    "mean": float(feature_series.mean()),
                    "std": float(feature_series.std()),
                    "min": float(feature_series.min()),
                    "max": float(feature_series.max()),
                }

        # 目标值统计
        self.target_stats = {}
        if self.target_column in self.indices_df.columns:
            target_series = self.indices_df[self.target_column]
            self.target_stats = {
                "mean": float(target_series.mean()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "median": float(target_series.median()),
            }

            # 计算四分位距
            q1 = float(target_series.quantile(0.25))
            q3 = float(target_series.quantile(0.75))
            self.target_stats["iqr"] = q3 - q1

            print(
                f"目标值统计: 均值={self.target_stats['mean']:.4f}, "
                f"标准差={self.target_stats['std']:.4f}, "
                f"范围=[{self.target_stats['min']:.4f}, {self.target_stats['max']:.4f}]"
            )

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_molecules

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        返回:
            Dictionary containing:
            - 'pos': 点坐标 (n_atoms, 3)
            - 'x': 节点特征 (n_atoms, n_features)
            - 'y': 目标值 (1,)
            - 'batch': 批处理索引 (n_atoms,)
            - 'num_atoms': 原子数 (1,)
        """
        # 获取分子ID
        mol_id = self.molecule_ids[idx]

        # 获取该分子的所有点
        molecule_points = self.points_df[self.points_df["molecule_id"] == mol_id]

        # 获取原子坐标
        pos = molecule_points[["x", "y", "z"]].values.astype(np.float32)

        # 获取节点特征
        if self.node_features:
            x_values = []
            for feature in self.node_features:
                if feature in molecule_points.columns:
                    # 确保特征值是数值类型
                    feature_values = safe_convert_to_numeric(
                        molecule_points[feature]
                    ).values.astype(np.float32)
                    x_values.append(feature_values)
            if x_values:
                x = np.column_stack(x_values)
            else:
                x = np.zeros((len(pos), 0), dtype=np.float32)
        else:
            x = np.zeros((len(pos), 0), dtype=np.float32)

        # 获取目标值
        y = np.array(
            [self.indices_df.loc[mol_id, self.target_column]], dtype=np.float32
        )

        # 获取原子数
        num_atoms = np.array([len(pos)], dtype=np.int64)

        # 转换为张量
        sample = {
            "pos": torch.tensor(pos, dtype=torch.float32),
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
        }

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_molecule_by_id(self, mol_id: int) -> Dict[str, torch.Tensor]:
        """通过分子ID获取分子数据"""
        idx = self.molecule_ids.index(mol_id)
        return self[idx]

    def get_molecule_smiles(self, mol_id: int) -> Optional[str]:
        """获取分子的SMILES表示（如果有）"""
        if "smiles" in self.indices_df.columns:
            return self.indices_df.loc[mol_id, "smiles"]
        return None


class PointCloudCollater:
    """
    点云数据批处理collator
    处理不同大小点云的批次
    """

    def __init__(self, follow_batch: List[str] = None):
        """
        初始化collator

        参数:
            follow_batch: 需要跟踪批次索引的键
        """
        self.follow_batch = follow_batch or []

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本合并成一个批次

        参数:
            batch: 样本列表

        返回:
            批处理后的字典
        """
        batch_dict = {}

        # 处理每种类型的张量
        for key in batch[0].keys():
            if key == "num_atoms":
                # 原子数直接堆叠
                batch_dict[key] = torch.cat([item[key] for item in batch], dim=0)
            elif key == "y":
                # 目标值堆叠
                batch_dict[key] = torch.cat([item[key] for item in batch], dim=0)
            elif key in ["pos", "x"]:
                # 坐标和特征拼接，并创建批次索引
                values = [item[key] for item in batch]
                batch_dict[key] = torch.cat(values, dim=0)

                # 创建批次索引
                batch_indices = []
                for i, val in enumerate(values):
                    batch_indices.extend([i] * len(val))
                batch_dict[f"{key}_batch"] = torch.tensor(
                    batch_indices, dtype=torch.long
                )

        # 计算总原子数
        batch_dict["total_atoms"] = torch.sum(batch_dict["num_atoms"])

        return batch_dict


class PointCloudTransform:
    """点云数据变换"""

    def __init__(
        self,
        normalize_pos: bool = False,
        normalize_features: bool = False,
        normalize_target: bool = False,
        feature_stats: Dict = None,
        target_stats: Dict = None,
        center_pos: bool = True,
        random_rotate: bool = False,
        target_key: str = "y",  # 目标值的字典键
        target_normalization_method: str = "zscore",  # 归一化方法: "zscore", "minmax", "robust"
        preserve_target_range: bool = False,  # 是否保留目标原始范围（仅对minmax有效）
    ):
        """
        初始化变换

        参数:
            normalize_pos: 是否标准化坐标
            normalize_features: 是否标准化特征
            normalize_target: 是否标准化目标值
            feature_stats: 特征统计信息
            target_stats: 目标统计信息
            center_pos: 是否将点云中心置于原点
            random_rotate: 是否随机旋转（数据增强）
            target_key: 目标值在字典中的键
            target_normalization_method: 目标归一化方法
            preserve_target_range: 是否保留目标原始范围（用于反向变换）
        """
        self.normalize_pos = normalize_pos
        self.normalize_features = normalize_features
        self.normalize_target = normalize_target
        self.feature_stats = feature_stats or {}
        self.target_stats = target_stats or {}
        self.center_pos = center_pos
        self.random_rotate = random_rotate
        self.target_key = target_key
        self.target_normalization_method = target_normalization_method
        self.preserve_target_range = preserve_target_range

        # 验证归一化方法
        valid_methods = ["zscore", "minmax", "robust", "log", "quantile"]
        if target_normalization_method not in valid_methods:
            raise ValueError(
                f"无效的归一化方法: {target_normalization_method}. 可选方法: {valid_methods}"
            )

    def __call__(self, sample: Dict) -> Dict:
        """应用变换"""
        # 一个样本是一个点云
        pos = sample["pos"]  # 位置 x, y, z
        x = sample["x"]

        # 1. 中心化点云
        if self.center_pos:
            pos_mean = pos.mean(dim=0, keepdim=True)
            pos = pos - pos_mean

        # 2. 标准化坐标
        if self.normalize_pos:
            pos_std = pos.std(dim=0, keepdim=True)
            pos_std = torch.where(pos_std == 0, torch.ones_like(pos_std), pos_std)
            pos = pos / pos_std

        # 3. 标准化特征
        if self.normalize_features and self.feature_stats:
            for i, feature in enumerate(self.feature_stats.keys()):
                if i < x.shape[1]:  # 确保索引在范围内
                    stats = self.feature_stats[feature]
                    if "mean" in stats and "std" in stats:
                        mean = torch.tensor(stats["mean"], dtype=torch.float32)
                        std = torch.tensor(stats["std"], dtype=torch.float32)
                        if std == 0:
                            std = torch.tensor(1.0, dtype=torch.float32)
                        x[:, i] = (x[:, i] - mean) / std

        # 4. 标准化目标值（如果存在）
        if self.normalize_target and self.target_key in sample:
            target = sample[self.target_key]
            stats = self.target_stats

            if self.target_normalization_method == "zscore":
                # Z-score标准化: (x - mean) / std
                if "mean" in stats and "std" in stats:
                    mean = torch.tensor(stats["mean"], dtype=torch.float32)
                    std = torch.tensor(stats["std"], dtype=torch.float32)
                    if std == 0:
                        std = torch.tensor(1.0, dtype=torch.float32)
                    target = (target - mean) / std

            elif self.target_normalization_method == "minmax":
                # 最小-最大归一化: (x - min) / (max - min)
                if "min" in stats and "max" in stats:
                    min_val = torch.tensor(stats["min"], dtype=torch.float32)
                    max_val = torch.tensor(stats["max"], dtype=torch.float32)
                    range_val = max_val - min_val
                    if range_val == 0:
                        range_val = torch.tensor(1.0, dtype=torch.float32)
                    target = (target - min_val) / range_val

                    # 可选：缩放到[0, 1]范围之外
                    if "target_min" in stats and "target_max" in stats:
                        new_min = torch.tensor(stats["target_min"], dtype=torch.float32)
                        new_max = torch.tensor(stats["target_max"], dtype=torch.float32)
                        target = target * (new_max - new_min) + new_min

            elif self.target_normalization_method == "robust":
                # 鲁棒归一化: (x - median) / IQR
                if "median" in stats and "iqr" in stats:
                    median = torch.tensor(stats["median"], dtype=torch.float32)
                    iqr = torch.tensor(stats["iqr"], dtype=torch.float32)
                    if iqr == 0:
                        iqr = torch.tensor(1.0, dtype=torch.float32)
                    target = (target - median) / iqr

            elif self.target_normalization_method == "log":
                # 对数变换: log(x + eps)
                eps = stats.get("eps", 1e-8)
                if "shift" in stats:  # 如果有偏移量
                    target = target + stats["shift"]
                target = torch.log(target + eps)

            elif self.target_normalization_method == "quantile":
                # 分位数归一化（需要预计算的分位数映射）
                if "quantile_mapping" in stats:
                    # 这里简化为使用预计算的均值和标准差
                    if "mean" in stats and "std" in stats:
                        mean = torch.tensor(stats["mean"], dtype=torch.float32)
                        std = torch.tensor(stats["std"], dtype=torch.float32)
                        if std == 0:
                            std = torch.tensor(1.0, dtype=torch.float32)
                        target = (target - mean) / std

            # 保存原始目标值以便反向变换
            if self.preserve_target_range:
                sample[f"{self.target_key}_original"] = sample[self.target_key].clone()

            # 更新归一化后的目标值
            sample[self.target_key] = target

        # 5. 随机旋转（数据增强）
        if self.random_rotate and torch.rand(1) > 0.5:
            # 生成随机旋转矩阵
            angle = torch.rand(1) * 2 * torch.pi
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)

            # 2D旋转（绕z轴）
            rotation_matrix = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
            )

            pos = torch.matmul(pos, rotation_matrix.T)

        sample["pos"] = pos
        sample["x"] = x

        return sample

    def inverse_transform_target(self, normalized_target: torch.Tensor) -> torch.Tensor:
        """将归一化的目标值反向变换回原始尺度"""
        if not self.normalize_target or not self.target_stats:
            return normalized_target

        stats = self.target_stats

        if self.target_normalization_method == "zscore":
            if "mean" in stats and "std" in stats:
                mean = torch.tensor(stats["mean"], dtype=torch.float32)
                std = torch.tensor(stats["std"], dtype=torch.float32)
                if std == 0:
                    std = torch.tensor(1.0, dtype=torch.float32)
                return normalized_target * std + mean

        elif self.target_normalization_method == "minmax":
            if "min" in stats and "max" in stats:
                min_val = torch.tensor(stats["min"], dtype=torch.float32)
                max_val = torch.tensor(stats["max"], dtype=torch.float32)
                range_val = max_val - min_val
                if range_val == 0:
                    range_val = torch.tensor(1.0, dtype=torch.float32)

                # 如果缩放过范围
                if "target_min" in stats and "target_max" in stats:
                    new_min = torch.tensor(stats["target_min"], dtype=torch.float32)
                    new_max = torch.tensor(stats["target_max"], dtype=torch.float32)
                    # 先缩放到[0, 1]
                    normalized_target = (normalized_target - new_min) / (
                        new_max - new_min
                    )

                return normalized_target * range_val + min_val

        elif self.target_normalization_method == "robust":
            if "median" in stats and "iqr" in stats:
                median = torch.tensor(stats["median"], dtype=torch.float32)
                iqr = torch.tensor(stats["iqr"], dtype=torch.float32)
                if iqr == 0:
                    iqr = torch.tensor(1.0, dtype=torch.float32)
                return normalized_target * iqr + median

        elif self.target_normalization_method == "log":
            eps = stats.get("eps", 1e-8)
            result = torch.exp(normalized_target) - eps
            if "shift" in stats:
                result = result - stats["shift"]
            return result

        elif self.target_normalization_method == "quantile":
            # 分位数归一化的逆变换较复杂，这里简化为Z-score逆变换
            if "mean" in stats and "std" in stats:
                mean = torch.tensor(stats["mean"], dtype=torch.float32)
                std = torch.tensor(stats["std"], dtype=torch.float32)
                if std == 0:
                    std = torch.tensor(1.0, dtype=torch.float32)
                return normalized_target * std + mean

        # 如果无法逆变换，返回原始值
        return normalized_target

    @staticmethod
    def compute_target_stats(
        targets: List[torch.Tensor], method: str = "zscore", **kwargs
    ) -> Dict:
        """
        计算目标值的统计信息

        参数:
            targets: 目标值列表
            method: 归一化方法
            **kwargs: 其他参数
        """
        if not targets:
            return {}

        # 合并所有目标值
        all_targets = torch.cat(targets)

        stats = {}

        if method == "zscore":
            stats["mean"] = float(all_targets.mean().item())
            stats["std"] = float(all_targets.std().item())

        elif method == "minmax":
            stats["min"] = float(all_targets.min().item())
            stats["max"] = float(all_targets.max().item())

            # 可选的自定义范围
            target_min = kwargs.get("target_min", 0.0)
            target_max = kwargs.get("target_max", 1.0)
            if target_min != 0.0 or target_max != 1.0:
                stats["target_min"] = target_min
                stats["target_max"] = target_max

        elif method == "robust":
            stats["median"] = float(torch.median(all_targets).item())

            # 计算四分位距
            q75, q25 = (
                torch.quantile(all_targets, 0.75),
                torch.quantile(all_targets, 0.25),
            )
            stats["iqr"] = float((q75 - q25).item())

        elif method == "log":
            # 确保所有值都是正数
            min_val = all_targets.min().item()
            if min_val <= 0:
                shift = abs(min_val) + kwargs.get("eps", 1e-8)
                stats["shift"] = shift
                all_targets = all_targets + shift
            stats["eps"] = kwargs.get("eps", 1e-8)

        elif method == "quantile":
            # 计算分位数映射（简化版本）
            stats["mean"] = float(all_targets.mean().item())
            stats["std"] = float(all_targets.std().item())

            # 实际应用中可能需要更复杂的分位数映射
            quantiles = kwargs.get("quantiles", 100)
            stats["quantiles"] = torch.quantile(
                all_targets, torch.linspace(0, 1, quantiles)
            ).tolist()

        return stats


def compute_dataset_target_stats(
    dataset: PointCloudQM9Dataset, target_column: str = None
) -> Dict:
    """
    计算数据集目标列的统计信息

    参数:
        dataset: PointCloudQM9Dataset实例
        target_column: 目标列名，如果为None则使用数据集中的target_column

    返回:
        目标统计信息字典
    """
    if target_column is None:
        target_column = dataset.target_column

    if target_column not in dataset.indices_df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据集中")

    target_series = dataset.indices_df[target_column]

    stats = {
        "mean": float(target_series.mean()),
        "std": float(target_series.std()),
        "min": float(target_series.min()),
        "max": float(target_series.max()),
        "median": float(target_series.median()),
    }

    # 计算四分位距
    q1 = float(target_series.quantile(0.25))
    q3 = float(target_series.quantile(0.75))
    stats["iqr"] = q3 - q1

    return stats


# 使用示例
if __name__ == "__main__":
    # 数据集路径
    points_csv = "./qm9.csv"
    indices_csv = "./qm9_indices.csv"

    # 创建完整数据集（不归一化目标值）
    print("创建数据集...")
    full_dataset = PointCloudQM9Dataset(
        points_csv=points_csv,
        indices_csv=indices_csv,
        transform=None,  # 先不应用变换
        target_column="internal_energy",
        node_features=[
            "atom_mass",
            "atom_valence_electrons",
            "atom_radius",
            "atom_mulliken_charge",
        ],
    )

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 从训练集计算目标值统计信息（推荐做法）
    print("\n计算训练集目标值统计信息...")
    train_targets = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_targets.append(sample["y"])

    train_target_stats = PointCloudTransform.compute_target_stats(
        train_targets, method="zscore"
    )
    print(f"训练集目标统计: {train_target_stats}")

    # 创建transform，使用训练集的统计信息
    transform = PointCloudTransform(
        normalize_pos=False,
        center_pos=True,
        random_rotate=False,
        normalize_target=True,
        target_stats=train_target_stats,
        target_normalization_method="zscore",
        preserve_target_range=True,  # 保存原始值以便反向变换
    )

    # 将transform应用到数据集
    full_dataset.transform = transform

    # 重新获取变换后的数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建collator
    collater = PointCloudCollater(follow_batch=["pos", "x"])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    # 测试数据加载器
    print("\n测试数据加载器...")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  位置形状: {batch['pos'].shape}")  # (总原子数, 3)
        print(f"  特征形状: {batch['x'].shape}")  # (总原子数, n_features)
        print(f"  批次索引形状: {batch['pos_batch'].shape}")
        print(f"  目标形状: {batch['y'].shape}")  # (batch_size, 1)
        print(f"  原子数: {batch['num_atoms'].shape}")  # (batch_size,)
        print(f"  总原子数: {batch['total_atoms'].item()}")

        # 显示归一化后的目标值范围
        y_min = batch["y"].min().item()
        y_max = batch["y"].max().item()
        print(f"  归一化目标值范围: [{y_min:.4f}, {y_max:.4f}]")

        if batch_idx >= 2:  # 只查看前3个批次
            break

    print(f"\n数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    print(
        f"目标值统计信息: 均值={train_target_stats.get('mean', 'N/A'):.4f}, "
        f"标准差={train_target_stats.get('std', 'N/A'):.4f}"
    )
