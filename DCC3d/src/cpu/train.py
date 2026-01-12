import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.model_selection import KFold
import logging
import time
import os
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 假设这些模块已正确导入
from DCC3d.src.cpu.data.dataset import OptimizedQM9Dataset, OptimizedQM9Collater
from module import DCConvNet


def get_dataloader_cv(
    n_splits: int = 5,
    batch_size: int = 32,
    val_batch_size: int = 16,
    random_state: int = 42,
    normalize_target: bool = False,
    fold_index: int = None,
):
    """
    使用5折交叉验证创建数据加载器

    参数:
        n_splits: 交叉验证折数
        batch_size: 训练批次大小
        val_batch_size: 验证/测试批次大小
        random_state: 随机种子
        normalize_target: 是否归一化目标值
        fold_index: 指定折索引（0到n_splits-1），如果为None则返回生成器

    返回:
        如果fold_index为None: 返回生成器，每次迭代返回(折索引, train_loader, val_loader, test_loader)
        如果fold_index指定: 返回指定折的(train_loader, val_loader, test_loader)
    """
    # 创建数据集
    dataset = OptimizedQM9Dataset(
        atoms_csv_path="./data/QM9/qm9_atoms.csv",
        energy_csv_path="./data/QM9/qm9_energy.csv",
        target_column="atomization_energy",
        normalize_mass_charge=True,
        normalize_pos=False,
        center_pos=True,
        cache_dir="./qm9_cache",
        force_reload=False,
        normalize_target=normalize_target,
        min_atoms=20,
    )

    # 创建collator
    collater = OptimizedQM9Collater()

    logger.info(f"数据集总大小: {len(dataset)}")
    logger.info(f"使用 {n_splits} 折交叉验证")

    # 记录数据集统计信息到wandb
    wandb.log(
        {
            "dataset_size": len(dataset),
            "n_splits": n_splits,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size,
        }
    )

    # 准备索引
    indices = np.arange(len(dataset))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def create_loaders_for_fold(train_idx, val_idx, fold_num):
        """为当前折创建数据加载器"""
        # 创建训练集和验证集的子集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 从训练集中划分出测试集（10%的训练集）
        from torch.utils.data import random_split

        train_size = int(0.9 * len(train_subset))
        test_size = len(train_subset) - train_size

        train_subset, test_subset = random_split(
            train_subset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(random_state),
        )

        logger.info(f"  训练集: {len(train_subset)} 个样本")
        logger.info(f"  验证集: {len(val_subset)} 个样本")
        logger.info(f"  测试集: {len(test_subset)} 个样本")

        # 记录数据划分信息到wandb
        wandb.log(
            {
                f"fold_{fold_num}_train_size": len(train_subset),
                f"fold_{fold_num}_val_size": len(val_subset),
                f"fold_{fold_num}_test_size": len(test_subset),
            }
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collater,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    # 如果指定了折索引，返回该折的数据
    if fold_index is not None:
        if fold_index < 0 or fold_index >= n_splits:
            raise ValueError(f"fold_index必须在0到{n_splits - 1}之间")

        all_splits = list(kfold.split(indices))
        train_idx, val_idx = all_splits[fold_index]

        logger.info(f"正在创建第 {fold_index + 1}/{n_splits} 折数据加载器...")
        return create_loaders_for_fold(train_idx, val_idx, fold_index + 1)

    # 返回生成器遍历所有折
    def fold_generator():
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"\n=== 第 {fold + 1}/{n_splits} 折 ===")

            train_loader, val_loader, test_loader = create_loaders_for_fold(
                train_idx, val_idx, fold + 1
            )

            yield fold, train_loader, val_loader, test_loader

    return fold_generator()


def evaluate(model, dataloader, criterion, device, dataloader_type="val"):
    """评估模型性能"""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    batch_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 注意：OptimizedQM9Collater移除了'pos_batch'，只有'batch'
            position_matrix = batch["pos"].to(device)
            channel_matrix = batch["x"].to(device)
            belonging = batch["batch"].to(device)  # 使用'batch'而不是'pos_batch'
            target = batch["y"].to(device)

            outputs = model(position_matrix, channel_matrix, belonging)
            loss = criterion(outputs, target)

            total_loss += loss.item()
            batch_losses.append(loss.item())
            all_targets.append(target.cpu())
            all_predictions.append(outputs.cpu())

    avg_loss = total_loss / len(dataloader)

    # 收集所有预测和目标值
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # 计算额外指标
    mae = torch.mean(torch.abs(all_predictions - all_targets))
    mse = torch.mean((all_predictions - all_targets) ** 2)
    rmse = torch.sqrt(mse)

    # 计算R²分数
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 计算相对误差
    relative_error = torch.mean(
        torch.abs(all_predictions - all_targets) / (torch.abs(all_targets) + 1e-8)
    )

    return {
        f"{dataloader_type}_loss": avg_loss,
        f"{dataloader_type}_mae": mae.item(),
        f"{dataloader_type}_mse": mse.item(),
        f"{dataloader_type}_rmse": rmse.item(),
        f"{dataloader_type}_r2": r2.item(),
        f"{dataloader_type}_relative_error": relative_error.item(),
        f"{dataloader_type}_batch_losses": batch_losses,
        f"{dataloader_type}_predictions": all_predictions.numpy(),
        f"{dataloader_type}_targets": all_targets.numpy(),
    }


def train_fold(
    fold_idx,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    device,
    project_name="DCConv3d_Energy_Prediction",
):
    """
    训练单个折的模型
    """
    # 为每一折创建单独的wandb运行
    wandb.init(
        project=project_name,
        name=f"fold_{fold_idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "fold": fold_idx + 1,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "val_batch_size": val_loader.batch_size,
            "device": str(device),
            "model_type": "DCConvNet",
            "loss_function": "L1Loss",
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        },
        tags=["cross-validation", "QM9", "energy-prediction", f"fold-{fold_idx + 1}"],
    )

    # 记录数据加载器信息
    wandb.log(
        {
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        }
    )

    # 初始化模型 - 注意输入特征数为2 (mass, charge)
    model = DCConvNet(num_features=2).to(device)

    # 记录模型架构和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    wandb.log(
        {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_architecture": str(model),
        }
    )

    # 使用MAE作为损失函数
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 记录梯度统计
    wandb.watch(model, criterion, log="all", log_freq=100)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 训练循环
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    train_history = []
    val_history = []

    # 训练计时
    fold_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batch_losses = []
        grad_norms = []

        train_bar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{num_epochs} [Train]",
        )

        for batch_idx, batch in enumerate(train_bar):
            batch_start_time = time.time()

            position_matrix = batch["pos"].to(device)
            channel_matrix = batch["x"].to(device)
            belonging = batch["batch"].to(device)
            target = batch["y"].to(device)

            # 确保target形状正确
            if target.dim() == 1:
                target = target.unsqueeze(1)

            # 前向传播
            outputs = model(position_matrix, channel_matrix, belonging)
            loss = criterion(outputs, target)

            # 添加 L1 正则化损失
            # l1_reg_loss = model.get_l1_regularization_loss()
            # if (l1_reg_loss >= loss):
            #     total_loss = loss + l1_reg_loss
            # else:
            total_loss = loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 计算梯度范数
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batch_losses.append(loss.item())

            batch_time = time.time() - batch_start_time

            # 每100个batch记录一次
            if batch_idx % 100 == 0:
                wandb.log(
                    {
                        f"fold_{fold_idx + 1}_batch_train_loss": loss.item(),
                        f"fold_{fold_idx + 1}_batch_grad_norm": total_norm,
                        f"fold_{fold_idx + 1}_batch_time": batch_time,
                        "batch": epoch * len(train_loader) + batch_idx,
                    }
                )

            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0

        # 验证阶段
        val_metrics = evaluate(model, val_loader, criterion, device, "val")
        val_loss = val_metrics["val_loss"]

        # 记录验证指标
        wandb.log(val_metrics)

        # 学习率调整
        scheduler.step(val_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time

        # 记录训练指标到wandb
        epoch_metrics = {
            "fold": fold_idx + 1,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
            "avg_grad_norm": avg_grad_norm,
            "train_batch_losses": train_batch_losses,
        }

        # 添加验证指标
        epoch_metrics.update(
            {
                k: v
                for k, v in val_metrics.items()
                if not k.endswith("_predictions")
                and not k.endswith("_targets")
                and not k.endswith("_batch_losses")
            }
        )

        wandb.log(epoch_metrics)

        # 保存历史记录
        train_history.append(avg_train_loss)
        val_history.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "fold": fold_idx + 1,
                "train_history": train_history,
                "val_history": val_history,
            }

            # 立即保存最佳模型
            torch.save(
                best_model_state,
                f"./train_data/checkpoint/best_model_fold_{fold_idx + 1}.pth",
            )

            # 打印学习率调整信息
            logger.info(f"Fold {fold_idx + 1} | 学习率调整为: {current_lr:.6f}")

            # 记录最佳模型指标
            wandb.log(
                {
                    f"fold_{fold_idx + 1}_best_val_loss": best_val_loss,
                    f"fold_{fold_idx + 1}_best_epoch": best_epoch,
                }
            )

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Fold {fold_idx + 1} | Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch}) | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
            )

            # 记录详细进度
            wandb.log(
                {
                    f"fold_{fold_idx + 1}_detailed_train_loss": avg_train_loss,
                    f"fold_{fold_idx + 1}_detailed_val_loss": val_loss,
                    f"fold_{fold_idx + 1}_detailed_epoch": epoch + 1,
                }
            )

    # 计算总训练时间
    fold_total_time = time.time() - fold_start_time

    # 记录训练时间
    wandb.log(
        {
            f"fold_{fold_idx + 1}_total_training_time": fold_total_time,
            f"fold_{fold_idx + 1}_avg_epoch_time": fold_total_time / num_epochs,
        }
    )

    # 测试阶段 - 使用最佳模型
    if best_model_state is None:
        # 如果没有保存最佳模型，保存最后一个
        best_model_state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "fold": fold_idx + 1,
            "train_history": train_history,
            "val_history": val_history,
        }
        torch.save(
            best_model_state,
            f"./train_data/checkpoint/best_model_fold_{fold_idx + 1}.pth",
        )

    # 加载最佳模型进行测试
    checkpoint = torch.load(
        f"./train_data/checkpoint/best_model_fold_{fold_idx + 1}.pth"
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # 测试评估
    test_metrics = evaluate(model, test_loader, criterion, device, "test")
    test_loss = test_metrics["test_loss"]

    # 记录测试指标
    wandb.log(test_metrics)

    # 创建预测vs真实值的散点图
    predictions = test_metrics["test_predictions"]
    targets = test_metrics["test_targets"]

    # 保存预测数据供后续分析
    np.savez(
        f"./train_data/checkpoint/fold_{fold_idx + 1}_predictions.npz",
        predictions=predictions,
        targets=targets,
    )

    logger.info(f"\nFold {fold_idx + 1} 训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
    logger.info(f"测试损失: {test_loss:.4f}")
    logger.info(f"总训练时间: {fold_total_time:.2f}秒")

    # 记录最终结果
    final_metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "total_training_time": fold_total_time,
        "avg_epoch_time": fold_total_time / num_epochs,
    }

    # 添加测试指标
    for key, value in test_metrics.items():
        if (
            not key.endswith("_predictions")
            and not key.endswith("_targets")
            and not key.endswith("_batch_losses")
        ):
            final_metrics[key] = value

    wandb.log(final_metrics)

    # 创建训练历史图表数据
    wandb.log(
        {
            "train_history": train_history,
            "val_history": val_history,
        }
    )

    wandb.finish()

    return model, best_val_loss, test_loss, test_metrics


def train_cross_validation(
    num_epochs=1000,
    learning_rate=0.001,
    n_splits=5,
    device=None,
    project_name="DCConv3d_Energy_Prediction_CV",
):
    """
    5折交叉验证训练主函数
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")
    logger.info(
        f"训练配置: {n_splits}折交叉验证, {num_epochs}轮, 学习率{learning_rate}"
    )

    # 创建主wandb运行记录总体信息
    main_run = wandb.init(
        project=project_name,
        name=f"cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "n_splits": n_splits,
            "device": str(device),
            "total_start_time": datetime.now().isoformat(),
        },
        tags=["cross-validation-summary", "QM9", "energy-prediction"],
    )

    # 获取交叉验证数据生成器
    fold_generator = get_dataloader_cv(
        n_splits=n_splits,
        batch_size=16,
        val_batch_size=8,
        random_state=42,
        normalize_target=True,
    )

    # 存储每折的结果
    fold_results = []
    fold_test_metrics = []

    cv_start_time = time.time()

    # 遍历每一折
    for fold_idx, train_loader, val_loader, test_loader in fold_generator:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始训练第 {fold_idx + 1}/{n_splits} 折")
        logger.info(f"{'=' * 60}")

        fold_start_time = time.time()

        # 训练当前折
        try:
            model, best_val_loss, test_loss, test_metrics = train_fold(
                fold_idx=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                project_name=project_name,
            )

            fold_time = time.time() - fold_start_time

            fold_result = {
                "fold": fold_idx + 1,
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "training_time": fold_time,
                "model": model,
            }

            fold_results.append(fold_result)
            fold_test_metrics.append(test_metrics)

            # 记录单折结果到主运行
            wandb.log(
                {
                    f"cv_fold_{fold_idx + 1}_best_val_loss": best_val_loss,
                    f"cv_fold_{fold_idx + 1}_test_loss": test_loss,
                    f"cv_fold_{fold_idx + 1}_training_time": fold_time,
                    f"cv_fold_{fold_idx + 1}_test_mae": test_metrics.get("test_mae", 0),
                    f"cv_fold_{fold_idx + 1}_test_r2": test_metrics.get("test_r2", 0),
                }
            )

        except Exception as e:
            logger.error(f"训练第 {fold_idx + 1} 折时发生错误: {e}")
            import traceback

            traceback.print_exc()
            continue

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_results:
        logger.error("所有折的训练都失败了！")
        wandb.log({"cv_status": "failed"})
        wandb.finish()
        return None, None, None

    # 计算总时间
    cv_total_time = time.time() - cv_start_time

    # 计算平均性能
    val_losses = [r["best_val_loss"] for r in fold_results]
    test_losses = [r["test_loss"] for r in fold_results]
    test_maes = [m.get("test_mae", 0) for m in fold_test_metrics]
    test_r2s = [m.get("test_r2", 0) for m in fold_test_metrics]

    avg_val_loss = np.mean(val_losses)
    avg_test_loss = np.mean(test_losses)
    avg_test_mae = np.mean(test_maes)
    avg_test_r2 = np.mean(test_r2s)

    std_test_loss = np.std(test_losses)
    std_test_mae = np.std(test_maes)
    std_test_r2 = np.std(test_r2s)

    logger.info(f"\n{'=' * 60}")
    logger.info("5折交叉验证结果汇总:")
    logger.info(f"{'=' * 60}")

    for result in fold_results:
        logger.info(
            f"折 {result['fold']}: "
            f"最佳验证损失={result['best_val_loss']:.4f}, "
            f"测试损失={result['test_loss']:.4f}, "
            f"训练时间={result['training_time']:.2f}秒"
        )

    logger.info(f"\n平均验证损失: {avg_val_loss:.4f}")
    logger.info(f"平均测试损失: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    logger.info(f"平均测试MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
    logger.info(f"平均测试R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")
    logger.info(f"总训练时间: {cv_total_time:.2f}秒")

    # 保存所有折的结果
    results_summary = {
        "fold_results": fold_results,
        "avg_val_loss": avg_val_loss,
        "avg_test_loss": avg_test_loss,
        "avg_test_mae": avg_test_mae,
        "avg_test_r2": avg_test_r2,
        "std_test_loss": std_test_loss,
        "std_test_mae": std_test_mae,
        "std_test_r2": std_test_r2,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "total_training_time": cv_total_time,
    }

    torch.save(results_summary, "./train_data/checkpoint/cv_results_summary.pth")

    # 记录汇总结果到wandb
    cv_summary_metrics = {
        "cv_avg_val_loss": avg_val_loss,
        "cv_avg_test_loss": avg_test_loss,
        "cv_avg_test_mae": avg_test_mae,
        "cv_avg_test_r2": avg_test_r2,
        "cv_std_test_loss": std_test_loss,
        "cv_std_test_mae": std_test_mae,
        "cv_std_test_r2": std_test_r2,
        "cv_total_training_time": cv_total_time,
        "cv_completion_time": datetime.now().isoformat(),
        "cv_status": "completed",
    }

    # 添加每折的详细数据
    for i, result in enumerate(fold_results):
        cv_summary_metrics[f"cv_fold_{i + 1}_details"] = {
            "best_val_loss": float(result["best_val_loss"]),
            "test_loss": float(result["test_loss"]),
            "training_time": float(result["training_time"]),
        }

    wandb.log(cv_summary_metrics)

    # 创建性能比较表格
    fold_table_data = []
    for i, result in enumerate(fold_results):
        fold_table_data.append(
            [
                i + 1,
                result["best_val_loss"],
                result["test_loss"],
                fold_test_metrics[i].get("test_mae", 0),
                fold_test_metrics[i].get("test_r2", 0),
                result["training_time"],
            ]
        )

    fold_table = wandb.Table(
        columns=[
            "Fold",
            "Best Val Loss",
            "Test Loss",
            "Test MAE",
            "Test R²",
            "Training Time",
        ],
        data=fold_table_data,
    )

    wandb.log({"cv_fold_performance_table": fold_table})

    # 保存性能图表数据
    performance_data = {
        "folds": [r["fold"] for r in fold_results],
        "val_losses": val_losses,
        "test_losses": test_losses,
        "test_maes": test_maes,
        "test_r2s": test_r2s,
    }

    import json

    with open("./train_data/checkpoint/performance_data.json", "w") as f:
        json.dump(performance_data, f, indent=2)

    # 记录性能数据到wandb
    wandb.log({"performance_data": json.dumps(performance_data)})

    wandb.finish()

    return fold_results, avg_test_loss, std_test_loss


def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs("./train_data/checkpoint", exist_ok=True)

    # 记录系统信息
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(
                f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )

    # 训练模式选择
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = (
            input("选择训练模式: (1) 5折交叉验证 (2) 单次划分训练 [默认: 1]: ").strip()
            or "1"
        )

    # 训练参数
    num_epochs = 150
    learning_rate = 0.001

    # 从命令行参数获取配置
    if len(sys.argv) > 2:
        try:
            num_epochs = int(sys.argv[2])
        except:
            pass
    if len(sys.argv) > 3:
        try:
            learning_rate = float(sys.argv[3])
        except:
            pass

    if mode == "1":
        # 5折交叉验证训练
        logger.info("开始5折交叉验证训练:")
        logger.info(f"  轮数: {num_epochs}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            fold_results, avg_test_loss, std_test_loss = train_cross_validation(
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                n_splits=5,
                device=device,
            )

            if fold_results is not None:
                logger.info(
                    f"\n训练完成! 平均测试损失: {avg_test_loss:.4f} ± {std_test_loss:.4f}"
                )
                logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # 生成训练报告
                report = {
                    "训练模式": "5折交叉验证",
                    "训练轮数": num_epochs,
                    "学习率": learning_rate,
                    "平均测试损失": f"{avg_test_loss:.4f} ± {std_test_loss:.4f}",
                    "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "设备": str(device),
                }

                logger.info("\n训练报告:")
                for key, value in report.items():
                    logger.info(f"  {key}: {value}")

            else:
                logger.error("训练失败!")

        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            import traceback

            traceback.print_exc()

    else:
        # 单次划分训练
        logger.info("开始单次划分训练:")
        logger.info(f"  轮数: {num_epochs}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 初始化wandb运行
        wandb.init(
            project="DCConv3d_Energy_Prediction_Single",
            name=f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": 256,
                "val_batch_size": 64,
                "device": str(device),
            },
            tags=["single-split", "QM9", "energy-prediction"],
        )

        try:
            # 使用交叉验证函数获取单折数据
            train_loader, val_loader, test_loader = get_dataloader_cv(
                n_splits=5,
                batch_size=16,
                val_batch_size=8,
                random_state=42,
                normalize_target=True,
                fold_index=0,  # 只取第一折
            )

            # 初始化模型
            model = DCConvNet(num_features=2).to(device)

            # 记录模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            wandb.log(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_architecture": str(model),
                }
            )

            # 使用MAE损失
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)

            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

            # 监控模型
            wandb.watch(model, criterion, log="all", log_freq=100)

            best_val_loss = float("inf")
            best_model_state = None
            train_history = []
            val_history = []

            start_time = time.time()

            for epoch in range(num_epochs):
                # 训练
                model.train()
                train_loss = 0.0
                grad_norms = []

                train_bar = tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
                )
                for batch_idx, batch in enumerate(train_bar):
                    position_matrix = batch["pos"].to(device)
                    channel_matrix = batch["x"].to(device)
                    belonging = batch["batch"].to(device)
                    target = batch["y"].to(device)

                    if target.dim() == 1:
                        target = target.unsqueeze(1)

                    outputs = model(position_matrix, channel_matrix, belonging)
                    loss = criterion(outputs, target)

                    optimizer.zero_grad()
                    loss.backward()

                    # 计算梯度范数
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    grad_norms.append(total_norm)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    train_loss += loss.item()

                    # 记录批次信息
                    if batch_idx % 100 == 0:
                        wandb.log(
                            {
                                "batch_train_loss": loss.item(),
                                "batch_grad_norm": total_norm,
                                "global_batch": epoch * len(train_loader) + batch_idx,
                            }
                        )

                    train_bar.set_postfix(loss=loss.item())

                avg_train_loss = train_loss / len(train_loader)
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0

                # 验证
                val_metrics = evaluate(model, val_loader, criterion, device, "val")
                val_loss = val_metrics["val_loss"]

                # 记录验证指标
                wandb.log(val_metrics)

                # 学习率调整
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                # 记录训练指标
                epoch_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "avg_grad_norm": avg_grad_norm,
                }

                # 添加验证指标
                epoch_metrics.update(
                    {
                        k: v
                        for k, v in val_metrics.items()
                        if not k.endswith("_predictions")
                        and not k.endswith("_targets")
                        and not k.endswith("_batch_losses")
                    }
                )

                wandb.log(epoch_metrics)

                # 保存历史
                train_history.append(avg_train_loss)
                val_history.append(val_loss)

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "train_history": train_history,
                        "val_history": val_history,
                    }
                    torch.save(
                        best_model_state,
                        "./train_data/checkpoint/best_model_single.pth",
                    )

                    # 记录最佳模型信息
                    wandb.log(
                        {
                            "best_val_loss": best_val_loss,
                            "best_epoch": epoch + 1,
                        }
                    )

                # 打印进度
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Best Val Loss: {best_val_loss:.4f} | LR: {current_lr:.6f}"
                    )

            # 计算总时间
            total_time = time.time() - start_time

            # 测试
            if best_model_state is None:
                best_model_state = {
                    "epoch": num_epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                torch.save(
                    best_model_state, "./train_data/checkpoint/best_model_single.pth"
                )

            checkpoint = torch.load("./train_data/checkpoint/best_model_single.pth")
            model.load_state_dict(checkpoint["model_state_dict"])

            test_metrics = evaluate(model, test_loader, criterion, device, "test")
            test_loss = test_metrics["test_loss"]

            # 记录测试指标
            wandb.log(test_metrics)

            # 记录最终结果
            final_metrics = {
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "total_training_time": total_time,
                "avg_epoch_time": total_time / num_epochs,
                "train_history": train_history,
                "val_history": val_history,
            }

            # 添加测试指标
            for key, value in test_metrics.items():
                if (
                    not key.endswith("_predictions")
                    and not key.endswith("_targets")
                    and not key.endswith("_batch_losses")
                ):
                    final_metrics[key] = value

            wandb.log(final_metrics)

            logger.info("\n训练完成!")
            logger.info(f"最佳验证损失: {best_val_loss:.4f}")
            logger.info(f"测试损失: {test_loss:.4f}")
            logger.info(f"总训练时间: {total_time:.2f}秒")
            logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            import traceback

            traceback.print_exc()
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
