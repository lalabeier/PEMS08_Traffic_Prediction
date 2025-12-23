"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import STGCN, get_laplacian
from preprocess import load_raw_data, split_train_test, normalize, create_sequences

def prepare_data(data_dir='../data/PEMS08_raw/', seq_len=12, pred_len=1):
    """准备数据加载器"""
    data = load_raw_data(data_dir)
    train_data, test_data = split_train_test(data)
    train_norm, mean, std = normalize(train_data)
    test_norm, _, _ = normalize(test_data, mean, std)
    X_train, y_train = create_sequences(train_norm, seq_len, pred_len)
    X_test, y_test = create_sequences(test_norm, seq_len, pred_len)

    # 转换为torch张量
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)  # (samples, features, nodes, seq_len)
    y_train = torch.FloatTensor(y_train).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test = torch.FloatTensor(y_test).permute(0, 3, 1, 2)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, X_train.shape[2]  # num_nodes

def train_one_epoch(model, loader, optimizer, criterion, device, L, focus_on_flow=True, scaler=None):
    """训练一个epoch
    Args:
        focus_on_flow: 如果为True，只对流量特征（通道0）计算损失
        scaler: 混合精度缩放器，如果为None则不使用混合精度
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        # 混合精度前向传播
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(X_batch, L)
                if focus_on_flow:
                    loss = criterion(output[:, 0:1, :, :], y_batch[:, 0:1, :, :])
                else:
                    loss = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(X_batch, L)
            if focus_on_flow:
                loss = criterion(output[:, 0:1, :, :], y_batch[:, 0:1, :, :])
            else:
                loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, L, focus_on_flow=True, scaler=None):
    """评估模型
    Args:
        focus_on_flow: 如果为True，只对流量特征（通道0）计算损失
        scaler: 如果提供，则在混合精度上下文内执行前向传播（用于评估时保持精度一致）
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(X_batch, L)
                    if focus_on_flow:
                        loss = criterion(output[:, 0:1, :, :], y_batch[:, 0:1, :, :])
                    else:
                        loss = criterion(output, y_batch)
            else:
                output = model(X_batch, L)
                if focus_on_flow:
                    loss = criterion(output[:, 0:1, :, :], y_batch[:, 0:1, :, :])
                else:
                    loss = criterion(output, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def plot_curves(train_losses, val_losses, save_path='../results/figures/loss_curve.png'):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    # 超参数
    seq_len = 12
    pred_len = 1
    epochs = 50
    lr = 0.001
    hidden_channels = 64
    K = 3

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 数据
    train_loader, val_loader, num_nodes = prepare_data(seq_len=seq_len, pred_len=pred_len)

    # 拉普拉斯矩阵（示例：使用单位矩阵作为占位）
    adj = torch.eye(num_nodes)
    L = get_laplacian(adj).to(device)

    # 模型
    model = STGCN(num_nodes, in_channels=3, hidden_channels=hidden_channels, out_channels=3, K=K).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    # 只对流量特征计算损失（因为我们只关心流量预测）
    focus_on_flow = True
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, L, focus_on_flow=focus_on_flow)
        val_loss = evaluate(model, val_loader, criterion, device, L, focus_on_flow=focus_on_flow)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存模型
    model_dir = '../results/models/'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'stgcn_model.pth'))
    print(f"模型已保存至 {model_dir}")

    # 绘制损失曲线
    plot_curves(train_losses, val_losses)

if __name__ == '__main__':
    main()