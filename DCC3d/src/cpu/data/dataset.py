import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import pickle
import os


class OptimizedQM9Dataset(Dataset):
    """
    优化的QM9数据集处理类，显著提高加载速度

    主要优化：
    1. 预加载并缓存数据到内存
    2. 使用numpy数组替代DataFrame查询
    3. 预计算分子索引范围
    4. 预归一化数据
    5. 移除不必要的字段
    """

    # 原子元素到相对原子质量的映射 (g/mol)
    ATOMIC_MASSES = {
        "H": 1.00794,
        "C": 12.0107,
        "N": 14.0067,
        "O": 15.9994,
        "F": 18.9984032,
    }

    def __init__(
        self,
        atoms_csv_path: str = "qm9_atoms.csv",
        energy_csv_path: str = "qm9_energy.csv",
        target_column: str = "atomization_energy",
        device: str = "cpu",
        transform: Optional[callable] = None,
        normalize_mass_charge: bool = True,
        normalize_target: bool = True,  # 新增：控制target归一化
        normalize_pos: bool = False,
        center_pos: bool = True,
        cache_dir: str = "./qm9_cache",
        force_reload: bool = False,
        # 新增：原子数筛选参数
        min_atoms: int = None,
        max_atoms: int = None,
        atoms_range: Tuple[int, int] = None,  # 替代min_atoms和max_atoms的元组形式
    ):
        """
        初始化优化的QM9数据集

        参数:
            atoms_csv_path: 原子数据CSV路径
            energy_csv_path: 能量数据CSV路径
            target_column: 目标列名
            device: 设备
            transform: 数据变换
            normalize_mass_charge: 是否对mass和charge进行归一化
            normalize_target: 是否对目标值进行归一化
            normalize_pos: 是否对位置进行归一化
            center_pos: 是否将分子中心化
            cache_dir: 缓存目录
            force_reload: 是否强制重新加载
            min_atoms: 最小原子数（包含）
            max_atoms: 最大原子数（包含）
            atoms_range: 原子数范围元组 (min_atoms, max_atoms)
        """
        super().__init__()

        self.atoms_csv_path = atoms_csv_path
        self.energy_csv_path = energy_csv_path
        self.target_column = target_column
        self.device = device
        self.transform = transform
        self.normalize_mass_charge = normalize_mass_charge
        self.normalize_pos = normalize_pos
        self.center_pos = center_pos
        self.cache_dir = cache_dir
        self.normalize_target = normalize_target

        # 处理原子数筛选参数
        if atoms_range is not None:
            self.min_atoms_filter = atoms_range[0]
            self.max_atoms_filter = atoms_range[1]
        else:
            self.min_atoms_filter = min_atoms
            self.max_atoms_filter = max_atoms

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 缓存文件名 - 包含筛选条件
        cache_name = f"dataset_cache_{target_column}"
        if self.min_atoms_filter is not None:
            cache_name += f"_min{self.min_atoms_filter}"
        if self.max_atoms_filter is not None:
            cache_name += f"_max{self.max_atoms_filter}"
        cache_file = os.path.join(cache_dir, f"{cache_name}.pkl")

        # 尝试加载缓存
        if not force_reload and os.path.exists(cache_file):
            print(f"正在从缓存加载数据集: {cache_file}")
            try:
                self._load_from_cache(cache_file)
                print(f"缓存加载成功: {self.num_molecules} 个分子")
                print(
                    f"筛选条件: 原子数范围 [{self.min_atoms_filter}, {self.max_atoms_filter}]"
                )
                return
            except Exception as e:
                print(f"缓存加载失败: {e}，重新从CSV加载")

        # 从CSV加载并构建缓存
        print("正在从CSV文件加载并构建缓存...")
        self._load_and_process_data()
        self._save_to_cache(cache_file)

    def _load_and_process_data(self):
        """加载并处理数据，构建高效的数据结构"""
        print(f"正在加载原子数据: {self.atoms_csv_path}")
        atoms_df = pd.read_csv(self.atoms_csv_path)

        print(f"正在加载能量数据: {self.energy_csv_path}")
        energy_df = pd.read_csv(self.energy_csv_path)

        # 添加原子质量列
        atoms_df["mass"] = atoms_df["element"].map(self.ATOMIC_MASSES)

        # 处理未知元素
        unknown_mask = atoms_df["mass"].isna()
        if unknown_mask.any():
            default_mass = 12.0  # 默认碳原子质量
            atoms_df.loc[unknown_mask, "mass"] = default_mass
            unknown_elements = atoms_df.loc[unknown_mask, "element"].unique()
            print(f"警告: 为未知元素设置默认质量 {default_mass}: {unknown_elements}")

        # 按分子分组并计算每个分子的原子数
        print("正在计算每个分子的原子数...")
        atoms_by_molecule = atoms_df.groupby("mol_id")
        molecule_atom_counts = atoms_by_molecule.size()

        # 筛选分子
        print("正在筛选分子...")
        valid_molecule_ids = []

        for mol_id, atom_count in molecule_atom_counts.items():
            # 检查是否满足筛选条件
            if self.min_atoms_filter is not None and atom_count < self.min_atoms_filter:
                continue
            if self.max_atoms_filter is not None and atom_count > self.max_atoms_filter:
                continue
            valid_molecule_ids.append(mol_id)

        # 获取所有有效的分子ID并排序
        self.molecule_ids = sorted(valid_molecule_ids)
        self.num_molecules = len(self.molecule_ids)

        print(f"原始分子总数: {len(molecule_atom_counts)}")
        print(f"筛选后分子数: {self.num_molecules}")

        if self.min_atoms_filter is not None or self.max_atoms_filter is not None:
            print(
                f"筛选条件: 原子数范围 [{self.min_atoms_filter}, {self.max_atoms_filter}]"
            )

        if self.num_molecules == 0:
            raise ValueError("没有满足筛选条件的分子！请调整原子数范围。")

        # 预处理数据：转换为numpy数组并预计算统计信息
        print("正在预处理数据...")

        # 预计算每个分子的原子索引范围
        self.atom_ranges = []
        self.molecule_targets = []
        self.molecule_atom_counts = []

        # 构建索引映射，避免重复查询
        atoms_by_molecule = atoms_df.groupby("mol_id")

        # 收集所有位置和特征数据
        all_positions = []
        all_features = []

        for mol_id in self.molecule_ids:
            mol_atoms = atoms_by_molecule.get_group(mol_id)

            # 确保原子按索引排序
            mol_atoms = mol_atoms.sort_values("atom_index")

            # 收集位置
            pos = mol_atoms[["x", "y", "z"]].values.astype(np.float32)
            all_positions.append(pos)

            # 收集特征：mass和charge
            mass = mol_atoms["mass"].values.astype(np.float32).reshape(-1, 1)
            charge = mol_atoms["charge"].values.astype(np.float32).reshape(-1, 1)
            features = np.concatenate([mass, charge], axis=1)
            all_features.append(features)

            # 记录原子范围
            start_idx = len(all_positions) - 1
            atom_count = len(pos)
            self.atom_ranges.append((start_idx, atom_count))
            self.molecule_atom_counts.append(atom_count)

            # 获取目标值
            target_row = energy_df[energy_df["mol_id"] == mol_id]
            if not target_row.empty:
                target_val = target_row[self.target_column].iloc[0]
                if pd.isna(target_val):
                    target_val = 0.0  # 处理NaN值
                self.molecule_targets.append(float(target_val))
            else:
                self.molecule_targets.append(0.0)  # 默认值
                print(f"警告: 分子 {mol_id} 没有目标值")

        # 转换为numpy数组列表（已预先分分子）
        self.all_positions = all_positions
        self.all_features = all_features
        self.molecule_targets = np.array(self.molecule_targets, dtype=np.float32)
        self.molecule_atom_counts = np.array(self.molecule_atom_counts, dtype=np.int32)

        # 计算统计信息
        self._compute_statistics(atoms_df, self.molecule_targets)

        print(f"数据预处理完成: {self.num_molecules} 个分子")

    def _compute_statistics(self, atoms_df, targets):
        """计算统计信息"""
        print("正在计算统计信息...")

        # 位置统计 - 仅使用筛选后的分子
        all_positions = []
        for pos in self.all_positions:
            all_positions.append(pos)
        all_positions_array = np.vstack(all_positions)

        self.pos_mean = np.mean(all_positions_array, axis=0, dtype=np.float32)
        self.pos_std = np.std(all_positions_array, axis=0, dtype=np.float32)
        self.pos_std = np.where(self.pos_std == 0, 1.0, self.pos_std)

        # mass和charge统计（用于归一化）- 仅使用筛选后的分子
        all_mass = []
        all_charge = []
        for features in self.all_features:
            if features.shape[1] >= 1:
                all_mass.append(features[:, 0])
            if features.shape[1] >= 2:
                all_charge.append(features[:, 1])

        if all_mass:
            all_mass_array = np.concatenate(all_mass)
            self.mass_mean = float(np.mean(all_mass_array))
            self.mass_std = float(np.std(all_mass_array))
            self.mass_std = self.mass_std if self.mass_std != 0 else 1.0
        else:
            self.mass_mean = 0.0
            self.mass_std = 1.0

        if all_charge:
            all_charge_array = np.concatenate(all_charge)
            self.charge_mean = float(np.mean(all_charge_array))
            self.charge_std = float(np.std(all_charge_array))
            self.charge_std = self.charge_std if self.charge_std != 0 else 1.0
        else:
            self.charge_mean = 0.0
            self.charge_std = 1.0

        # 目标值统计
        self.target_mean = float(np.mean(targets))
        self.target_std = float(np.std(targets))
        self.target_std = self.target_std if self.target_std != 0 else 1.0

        # 分子大小统计
        self.max_atoms = int(self.molecule_atom_counts.max())
        self.min_atoms = int(self.molecule_atom_counts.min())
        self.avg_atoms = float(self.molecule_atom_counts.mean())

        print(f"位置统计: 均值={self.pos_mean}, 标准差={self.pos_std}")
        print(f"质量统计: 均值={self.mass_mean:.4f}, 标准差={self.mass_std:.4f}")
        print(f"电荷统计: 均值={self.charge_mean:.4f}, 标准差={self.charge_std:.4f}")
        print(f"目标值统计: 均值={self.target_mean:.4f}, 标准差={self.target_std:.4f}")
        print(
            f"分子大小: 最大={self.max_atoms}, 最小={self.min_atoms}, 平均={self.avg_atoms:.2f}"
        )

        # 预归一化数据（如果启用）
        if self.normalize_mass_charge:
            print("正在预归一化mass和charge特征...")
            for i in range(self.num_molecules):
                features = self.all_features[i]
                if features.shape[1] >= 1:
                    # 归一化mass
                    features[:, 0] = (features[:, 0] - self.mass_mean) / self.mass_std
                if features.shape[1] >= 2:
                    # 归一化charge
                    features[:, 1] = (
                        features[:, 1] - self.charge_mean
                    ) / self.charge_std
                    self.all_features[i] = features

        if self.normalize_pos:
            print("正在预归一化位置数据...")
            for i in range(self.num_molecules):
                pos = self.all_positions[i]
                pos = (pos - self.pos_mean) / self.pos_std
                self.all_positions[i] = pos

        # 仅在启用时归一化target
        if self.normalize_target:
            self.molecule_targets = (
                self.molecule_targets - self.target_mean
            ) / self.target_std
        else:
            # 如果不归一化，直接使用原始值
            pass  # self.molecule_targets 已经是原始值

    def _save_to_cache(self, cache_file: str):
        """保存处理后的数据到缓存"""
        print(f"正在保存数据到缓存: {cache_file}")

        cache_data = {
            "molecule_ids": self.molecule_ids,
            "num_molecules": self.num_molecules,
            "all_positions": self.all_positions,
            "all_features": self.all_features,
            "molecule_targets": self.molecule_targets,
            "molecule_atom_counts": self.molecule_atom_counts,
            "atom_ranges": self.atom_ranges,
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "mass_mean": self.mass_mean,
            "mass_std": self.mass_std,
            "charge_mean": self.charge_mean,
            "charge_std": self.charge_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "max_atoms": self.max_atoms,
            "min_atoms": self.min_atoms,
            "avg_atoms": self.avg_atoms,
            "normalize_mass_charge": self.normalize_mass_charge,
            "normalize_pos": self.normalize_pos,
            "center_pos": self.center_pos,
            "min_atoms_filter": self.min_atoms_filter,
            "max_atoms_filter": self.max_atoms_filter,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"缓存保存完成: {os.path.getsize(cache_file) / (1024 * 1024):.2f} MB")

    def _load_from_cache(self, cache_file: str):
        """从缓存加载数据"""
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        # 恢复所有属性
        for key, value in cache_data.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        """返回数据集中的分子数量"""
        return self.num_molecules

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个分子的数据（已优化，速度极快）

        返回:
            Dictionary containing:
            - 'pos': 原子坐标 (n_atoms, 3)
            - 'x': 节点特征 (n_atoms, 2) [mass, charge]
            - 'y': 目标值 (1,) [原子化能]
            - 'batch': 批次索引 (n_atoms,) [所有原子都属于同一个分子]
            - 'num_atoms': 原子数 (1,)
        """
        # 直接从预加载的数组中获取数据
        pos = self.all_positions[idx].copy()  # 使用copy避免原地修改
        x = self.all_features[idx].copy()
        y = self.molecule_targets[idx]
        num_atoms = self.molecule_atom_counts[idx]

        # 中心化位置（如果需要）
        if self.center_pos:
            pos = pos - pos.mean(axis=0, keepdims=True)

        # 创建批次索引
        batch = np.full((num_atoms,), idx, dtype=np.int64)

        # 转换为张量
        sample = {
            "pos": torch.tensor(pos, dtype=torch.float32),
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor([y], dtype=torch.float32),
            "batch": torch.tensor(batch, dtype=torch.long),
            "num_atoms": torch.tensor([num_atoms], dtype=torch.long),
        }

        # 应用额外的变换（如果有）
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_molecule_info(self, mol_id: str) -> Dict:
        """获取分子的详细信息"""
        if mol_id not in self.molecule_ids:
            raise ValueError(f"分子ID {mol_id} 不存在")

        idx = self.molecule_ids.index(mol_id)
        pos = self.all_positions[idx]
        features = self.all_features[idx]

        info = {
            "mol_id": mol_id,
            "num_atoms": self.molecule_atom_counts[idx],
            "positions": pos.tolist(),
            "mass": features[:, 0].tolist() if features.shape[1] > 0 else [],
            "charge": features[:, 1].tolist() if features.shape[1] > 1 else [],
            "target": float(self.molecule_targets[idx]),
        }

        return info

    def get_atom_count_distribution(self) -> Dict[int, int]:
        """
        获取原子数分布

        返回:
            字典，键为原子数，值为该原子数的分子数量
        """
        distribution = {}
        for count in self.molecule_atom_counts:
            distribution[count] = distribution.get(count, 0) + 1

        return dict(sorted(distribution.items()))

    def filter_by_atom_count(
        self, min_atoms: int, max_atoms: int
    ) -> "OptimizedQM9Dataset":
        """
        基于原子数筛选分子，返回新的数据集实例

        参数:
            min_atoms: 最小原子数
            max_atoms: 最大原子数

        返回:
            新的数据集实例
        """
        # 创建新的筛选范围
        new_min = (
            max(self.min_atoms_filter, min_atoms)
            if self.min_atoms_filter is not None
            else min_atoms
        )
        new_max = (
            min(self.max_atoms_filter, max_atoms)
            if self.max_atoms_filter is not None
            else max_atoms
        )

        # 创建新的数据集实例
        return OptimizedQM9Dataset(
            atoms_csv_path=self.atoms_csv_path,
            energy_csv_path=self.energy_csv_path,
            target_column=self.target_column,
            device=self.device,
            transform=self.transform,
            normalize_mass_charge=self.normalize_mass_charge,
            normalize_target=self.normalize_target,
            normalize_pos=self.normalize_pos,
            center_pos=self.center_pos,
            cache_dir=self.cache_dir,
            force_reload=False,  # 会尝试从缓存加载
            min_atoms=new_min,
            max_atoms=new_max,
        )


class OptimizedQM9Collater:
    """
    优化的QM9数据批处理collator
    移除了不必要的字段，只保留核心信息
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本合并成一个批次

        参数:
            batch: 样本列表

        返回:
            批处理后的字典，仅包含必要字段
        """
        batch_dict = {}

        # 处理位置坐标
        pos_values = [item["pos"] for item in batch]
        batch_dict["pos"] = torch.cat(pos_values, dim=0)

        # 处理特征
        x_values = [item["x"] for item in batch]
        batch_dict["x"] = torch.cat(x_values, dim=0)

        # 处理目标值
        y_values = [item["y"] for item in batch]
        batch_dict["y"] = torch.cat(y_values, dim=0)

        # 处理原子数
        num_atoms = [item["num_atoms"] for item in batch]
        batch_dict["num_atoms"] = torch.cat(num_atoms, dim=0)

        # 创建批次索引（这是必要的，用于区分不同分子）
        batch_indices = []
        for i, pos in enumerate(pos_values):
            batch_indices.extend([i] * len(pos))
        batch_dict["batch"] = torch.tensor(batch_indices, dtype=torch.long)

        return batch_dict


# 使用示例和性能测试
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    print("测试优化的QM9数据集...")

    # 计时开始
    start_time = time.time()

    # 示例1：创建包含20个原子以上分子的数据集
    print("\n=== 示例1：筛选20个原子以上的分子 ===")
    dataset_20plus = OptimizedQM9Dataset(
        atoms_csv_path="./QM9/qm9_atoms.csv",
        energy_csv_path="./QM9/qm9_energy.csv",
        target_column="atomization_energy",
        normalize_mass_charge=True,
        normalize_pos=False,
        center_pos=True,
        cache_dir="./qm9_cache",
        force_reload=False,
        min_atoms=20,  # 只包含20个原子以上的分子
        max_atoms=None,  # 不设上限
    )

    load_time = time.time() - start_time
    print(f"数据集加载时间: {load_time:.2f} 秒")
    print(f"筛选后分子数: {len(dataset_20plus)}")

    # 查看原子数分布
    distribution = dataset_20plus.get_atom_count_distribution()
    print(f"原子数分布: {distribution}")

    # 示例2：创建包含15-25个原子的分子数据集
    print("\n=== 示例2：筛选15-25个原子的分子 ===")
    dataset_15_25 = OptimizedQM9Dataset(
        atoms_csv_path="./QM9/qm9_atoms.csv",
        energy_csv_path="./QM9/qm9_energy.csv",
        target_column="atomization_energy",
        normalize_mass_charge=True,
        normalize_pos=False,
        center_pos=True,
        cache_dir="./qm9_cache",
        force_reload=False,
        min_atoms=15,
        max_atoms=25,
    )

    print(f"筛选后分子数: {len(dataset_15_25)}")
    distribution = dataset_15_25.get_atom_count_distribution()
    print(f"原子数分布: {distribution}")

    # 示例3：使用atoms_range参数
    print("\n=== 示例3：使用atoms_range参数 ===")
    dataset_range = OptimizedQM9Dataset(
        atoms_csv_path="./QM9/qm9_atoms.csv",
        energy_csv_path="./QM9/qm9_energy.csv",
        target_column="atomization_energy",
        normalize_mass_charge=True,
        normalize_pos=False,
        center_pos=True,
        cache_dir="./qm9_cache",
        force_reload=False,
        atoms_range=(20, 30),  # 使用元组形式指定范围
    )

    print(f"筛选后分子数: {len(dataset_range)}")
    distribution = dataset_range.get_atom_count_distribution()
    print(f"原子数分布: {distribution}")

    # 可视化原子数分布
    plt.figure(figsize=(10, 6))

    # 获取三个数据集的分布
    datasets = [
        ("20+ 原子", dataset_20plus),
        ("15-25 原子", dataset_15_25),
        ("20-30 原子", dataset_range),
    ]

    for i, (name, dataset) in enumerate(datasets):
        distribution = dataset.get_atom_count_distribution()
        atoms = list(distribution.keys())
        counts = list(distribution.values())

        plt.subplot(1, 3, i + 1)
        plt.bar(atoms, counts)
        plt.title(f"{name}\n总分子数: {len(dataset)}")
        plt.xlabel("原子数")
        plt.ylabel("分子数量")
        plt.tight_layout()

    plt.suptitle("不同筛选条件下的原子数分布", fontsize=14, y=1.02)
    plt.show()

    # 测试获取样本
    print("\n测试样本获取...")
    sample = dataset_20plus[0]
    print(f"样本原子数: {sample['num_atoms'].item()}")
    print(f"样本特征形状: {sample['x'].shape}")

    # 创建collator和数据加载器
    collater = OptimizedQM9Collater()

    # 划分数据集
    print("\n划分数据集...")
    total_size = len(dataset_20plus)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    from torch.utils.data import random_split

    train_dataset, val_dataset, test_dataset = random_split(
        dataset_20plus, [train_size, val_size, test_size]
    )

    print(f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")

    # 创建数据加载器并测试性能
    print("\n测试数据加载器性能...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    # 测试一个批次
    batch_start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  位置形状: {batch['pos'].shape}")
        print(f"  特征形状: {batch['x'].shape}")
        print(f"  批次索引形状: {batch['batch'].shape}")
        print(f"  目标形状: {batch['y'].shape}")
        print(f"  原子数: {batch['num_atoms'].shape}")

        # 计算批次中的平均原子数
        avg_atoms = batch["num_atoms"].float().mean().item()
        print(f"  批次平均原子数: {avg_atoms:.2f}")

        if batch_idx >= 2:  # 只查看前3个批次
            break

    batch_time = time.time() - batch_start
    print(f"\n批次处理时间: {batch_time:.2f} 秒")

    # 显示统计信息
    print("\n数据集统计信息:")
    print(f"  总分子数: {len(dataset_20plus)}")
    print(f"  筛选条件: 原子数 ≥ {dataset_20plus.min_atoms_filter}")
    print(f"  实际原子数范围: {dataset_20plus.min_atoms} - {dataset_20plus.max_atoms}")
    print(f"  平均原子数: {dataset_20plus.avg_atoms:.2f}")
    print(f"  是否对mass和charge归一化: {dataset_20plus.normalize_mass_charge}")
