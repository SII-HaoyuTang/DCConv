import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
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
    ):
        """
        初始化优化的QM9数据集

        参数:
            normalize_mass_charge: 是否对mass和charge进行归一化 (默认为True)
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

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 缓存文件名
        cache_file = os.path.join(cache_dir, f"dataset_cache_{target_column}.pkl")

        # 尝试加载缓存
        if not force_reload and os.path.exists(cache_file):
            print(f"正在从缓存加载数据集: {cache_file}")
            try:
                self._load_from_cache(cache_file)
                print(f"缓存加载成功: {self.num_molecules} 个分子")
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

        # 获取所有唯一的分子ID并排序
        self.molecule_ids = sorted(atoms_df["mol_id"].unique())
        self.num_molecules = len(self.molecule_ids)

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

        # 位置统计
        all_positions = atoms_df[["x", "y", "z"]].values
        self.pos_mean = np.mean(all_positions, axis=0, dtype=np.float32)
        self.pos_std = np.std(all_positions, axis=0, dtype=np.float32)
        self.pos_std = np.where(self.pos_std == 0, 1.0, self.pos_std)

        # mass和charge统计（用于归一化）
        self.mass_mean = float(atoms_df["mass"].mean())
        self.mass_std = float(atoms_df["mass"].std())
        self.mass_std = self.mass_std if self.mass_std != 0 else 1.0

        self.charge_mean = float(atoms_df["charge"].mean())
        self.charge_std = float(atoms_df["charge"].std())
        self.charge_std = self.charge_std if self.charge_std != 0 else 1.0

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
                if features.shape[1] >= 2:
                    # 归一化mass
                    features[:, 0] = (features[:, 0] - self.mass_mean) / self.mass_std
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
            self.molecule_targets = targets

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

    print("测试优化的QM9数据集...")

    # 计时开始
    start_time = time.time()

    # 创建数据集（首次运行会构建缓存，较慢）
    dataset = OptimizedQM9Dataset(
        atoms_csv_path="./QM9/qm9_atoms.csv",
        energy_csv_path="./QM9/qm9_energy.csv",
        target_column="atomization_energy",
        normalize_mass_charge=True,  # 对mass和charge进行归一化
        normalize_pos=False,
        center_pos=True,
        cache_dir="./qm9_cache",
        force_reload=False,  # 设为True可强制重新构建缓存
        normalize_target=False,
    )

    load_time = time.time() - start_time
    print(f"数据集加载时间: {load_time:.2f} 秒")

    # 测试获取样本的速度
    print("\n测试样本获取速度...")
    test_start = time.time()

    # 获取多个样本
    test_indices = list(range(min(100, len(dataset))))
    samples = []
    for idx in test_indices:
        sample = dataset[idx]
        samples.append(sample)

    sample_time = time.time() - test_start
    print(f"获取 {len(test_indices)} 个样本的时间: {sample_time:.2f} 秒")
    print(f"平均每个样本: {sample_time / len(test_indices) * 1000:.2f} 毫秒")

    # 检查样本结构
    sample = dataset[0]
    print("\n单个样本结构:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} | dtype: {value.dtype}")
        else:
            print(f"  {key}: {value}")

    # 检查mass和charge是否已归一化
    print("\n检查mass和charge归一化:")
    print(f"  mass均值: {dataset.mass_mean:.4f}, 标准差: {dataset.mass_std:.4f}")
    print(f"  charge均值: {dataset.charge_mean:.4f}, 标准差: {dataset.charge_std:.4f}")

    # 验证归一化效果
    if len(samples) > 0:
        all_mass = torch.cat([s["x"][:, 0] for s in samples])
        all_charge = torch.cat([s["x"][:, 1] for s in samples])

        print(
            f"  归一化后mass均值: {all_mass.mean():.4f}, 标准差: {all_mass.std():.4f}"
        )
        print(
            f"  归一化后charge均值: {all_charge.mean():.4f}, 标准差: {all_charge.std():.4f}"
        )

    # 创建collator和数据加载器
    collater = OptimizedQM9Collater()

    # 划分数据集
    print("\n划分数据集...")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    from torch.utils.data import random_split

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
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

        # 确认没有不必要的字段
        print(f"  批次中包含的键: {list(batch.keys())}")

        if batch_idx >= 2:  # 只查看前3个批次
            break

    batch_time = time.time() - batch_start
    print(f"\n批次处理时间: {batch_time:.2f} 秒")

    # 显示统计信息
    print("\n数据集统计信息:")
    print(f"  总分子数: {len(dataset)}")
    print(f"  最大原子数: {dataset.max_atoms}")
    print(f"  平均原子数: {dataset.avg_atoms:.2f}")
    print(f"  是否对mass和charge归一化: {dataset.normalize_mass_charge}")
