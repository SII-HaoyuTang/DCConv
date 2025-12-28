import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from DCC3d.src.cpu.data.dataset import PointCloudCollater, PointCloudQM9Dataset, PointCloudTransform
from module import DCConvNet

def get_dataloader():
    # 数据集路径
    points_csv = "./data/qm9.csv"
    indices_csv = "./data/qm9_indices.csv"

    # 创建变换
    transform = PointCloudTransform(
        normalize_pos=False,
        center_pos=True,
        random_rotate=False,  # 训练时开启数据增强
    )

    # 创建完整数据集
    full_dataset = PointCloudQM9Dataset(
        points_csv=points_csv,
        indices_csv=indices_csv,
        transform=transform,
        target_column="internal_energy",
        node_features=[
            "atom_mass",
            "atom_valence_electrons",
            "atom_radius",
            "atom_mulliken_charge",
        ],  # 根据实际列名调整
    )

    # 划分数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建collator
    collater = PointCloudCollater(follow_batch=["pos", "x"])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for batch in dataloader:
            position_matrix, channel_matrix, belonging, target = (
                batch["pos"].to(device),
                batch["x"].to(device),
                batch["pos_batch"].to(device),
                (batch['y']/batch['num_atoms']).to(device),
            )

            outputs = model(position_matrix, channel_matrix, belonging)
            loss = criterion(outputs, target)
            mae = torch.mean(torch.abs(outputs - target))

            total_loss += loss.item()
            total_mae += mae.item()

    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    return avg_loss, avg_mae

def train(num_epochs, learning_rate, device):
    # 初始化 wandb
    wandb.init(project="DCConv3d_Energy_Prediction")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloader()

    # 初始化模型
    model = DCConvNet(num_features=4).to(device)  # 假设输入特征数为4
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            position_matrix, channel_matrix, belonging, target = (
                batch['pos'].to(device),
                batch['x'].to(device),
                batch['pos_batch'].to(device),
                (batch['y']/batch['num_atoms']).to(device),
            )

            # 确保 target 的形状与 outputs 的形状一致
            if target.dim() == 1:
                target = target.unsqueeze(1)  # 将 target 从 [128] 调整为 [128, 1]

            # 前向传播
            outputs = model(position_matrix, channel_matrix, belonging)
            loss = criterion(outputs, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)

        # 验证
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)

        # 使用 wandb 记录损失和准确率
        wandb.log({
            "Train Loss": avg_train_loss,
            "Validation Loss": val_loss,
            "Validation MAE": val_mae
        })

        # 打印训练和验证损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

        # 保存模型检查点
        torch.save(model.state_dict(), "./train_data/checkpoint/checkpoint_epoch.pth")

    # 测试
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    wandb.log({
        "Test Loss": test_loss,
        "Test MAE": test_mae
    })

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    learning_rate = 0.001

    # 开始训练
    train(num_epochs, learning_rate, device)