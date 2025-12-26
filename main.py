"""
主程序 - 一键运行交通流量预测任务
"""

import argparse
import sys
import os
import config

sys.path.append('./code')

from preprocess import load_raw_data, split_train_test, normalize, create_sequences
from model import STGCN, get_laplacian
from train import train_one_epoch, evaluate, plot_curves, WeightedMSELoss, HuberLoss
from evaluate import evaluate_on_test
from utils import build_corr_adjacency, load_adjacency_from_file
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='公路车流量预测')
    parser.add_argument('--mode', type=str, default='all', choices=['preprocess', 'train', 'evaluate', 'all'],
                        help='运行模式: preprocess仅预处理, train仅训练, evaluate仅评估, all全部执行')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='原始数据目录')
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH, help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='批大小')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='学习率')
    parser.add_argument('--seq_len', type=int, default=config.SEQ_LEN, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=config.PRED_LEN, help='预测长度')
    parser.add_argument('--hidden_channels', type=int, default=config.HIDDEN_CHANNELS, help='隐藏通道数')
    parser.add_argument('--K', type=int, default=config.K, help='图卷积阶数')
    parser.add_argument('--seed', type=int, default=config.SEED, help='随机种子')
    # 邻接矩阵 / 图结构
    parser.add_argument('--adj_method', type=str, default=config.ADJ_METHOD, choices=['corr', 'file'],
                        help='邻接矩阵构建方法: corr基于相关性自动构建, file从文件加载')
    parser.add_argument('--adj_path', type=str, default=config.ADJ_PATH, help='邻接矩阵文件路径（npy或csv），adj_method=file时使用')
    parser.add_argument('--adj_topk', type=int, default=config.ADJ_TOPK, help='相关性图每个节点保留的最大邻居数')
    parser.add_argument('--adj_threshold', type=float, default=config.ADJ_THRESHOLD, help='相关性阈值，低于该值视为0')
    # 数据清洗
    parser.add_argument('--handle_missing', action='store_true', default=config.HANDLE_MISSING, help='是否处理缺失值')
    parser.add_argument('--handle_outliers', action='store_true', default=config.HANDLE_OUTLIERS, help='是否处理异常值')
    parser.add_argument('--smooth_low_flow', action='store_true', default=config.SMOOTH_LOW_FLOW, help='是否平滑低流量段')
    parser.add_argument('--outlier_method', type=str, default=config.OUTLIER_METHOD, choices=['cap', 'smooth', 'remove'],
                        help='异常值处理方法')
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess(data_dir, seq_len, pred_len, handle_missing=True, handle_outliers=True,
               smooth_low_flow=True, outlier_method='cap', add_time_features=True, time_features_for_input_only=True):
    """预处理数据
    
    Args:
        data_dir: 数据目录
        seq_len: 序列长度
        pred_len: 预测长度
        handle_missing: 是否处理缺失值
        handle_outliers: 是否处理异常值
        smooth_low_flow: 是否平滑低流量段
        outlier_method: 异常值处理方法 ('cap', 'smooth', 'remove')
        add_time_features: 是否添加时间特征（小时、星期几）
        time_features_for_input_only: 时间特征是否仅添加到输入序列（不添加到预测目标）
    """
    print("开始预处理...")
    data = load_raw_data(data_dir,
                        handle_missing=handle_missing,
                        handle_outliers_flag=handle_outliers,
                        smooth_low_flow_flag=smooth_low_flow,
                        outlier_method=outlier_method,
                        verbose=True)
    train_data, test_data = split_train_test(data)
    train_norm, mean, std = normalize(train_data)
    test_norm, _, _ = normalize(test_data, mean, std)
    X_train, y_train = create_sequences(train_norm, seq_len, pred_len,
                                        add_time_features_flag=add_time_features,
                                        time_features_for_input_only=time_features_for_input_only)
    X_test, y_test = create_sequences(test_norm, seq_len, pred_len,
                                      add_time_features_flag=add_time_features,
                                      time_features_for_input_only=time_features_for_input_only)
    print("预处理完成。")
    return X_train, y_train, X_test, y_test, mean, std

def build_adjacency(args, num_nodes):
    """根据配置构建邻接矩阵"""
    if args.adj_method == 'file':
        if not args.adj_path:
            raise ValueError("adj_method=file 时必须提供 --adj_path")
        adj_np = load_adjacency_from_file(args.adj_path, num_nodes=num_nodes)
    else:
        # 基于相关性自动构建，需要原始清洗后的数据
        data_clean = load_raw_data(args.data_dir,
                                   handle_missing=args.handle_missing,
                                   handle_outliers_flag=args.handle_outliers,
                                   smooth_low_flow_flag=args.smooth_low_flow,
                                   outlier_method=args.outlier_method,
                                   verbose=False)
        adj_np = build_corr_adjacency(data_clean,
                                      k=args.adj_topk,
                                      threshold=args.adj_threshold,
                                      use_abs=True)
    return torch.FloatTensor(adj_np)

def train_model(X_train, y_train, X_test, y_test, args):
    """训练模型"""
    print("开始训练...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 转换为张量
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_train = torch.FloatTensor(y_train).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test = torch.FloatTensor(y_test).permute(0, 3, 1, 2)

    num_nodes = X_train.shape[2]
    adj = build_adjacency(args, num_nodes).to(device)
    L = get_laplacian(adj).to(device)

    # 动态确定输入输出通道数
    in_channels = X_train.shape[1]  # 特征维度在permute后变为索引1
    out_channels = y_train.shape[1]
    print(f"输入通道数: {in_channels}, 输出通道数: {out_channels}")

    model = STGCN(num_nodes, in_channels=in_channels, hidden_channels=args.hidden_channels, out_channels=out_channels, K=args.K).to(device)
    # 根据配置选择损失函数
    if config.LOSS == 'WeightedMSE' and config.USE_WEIGHTED_LOSS:
        criterion = WeightedMSELoss(
            low_threshold=config.LOW_FLOW_THRESHOLD,
            high_threshold=config.HIGH_FLOW_THRESHOLD,
            low_weight=config.LOW_FLOW_WEIGHT,
            high_weight=config.HIGH_FLOW_WEIGHT,
            normal_weight=config.NORMAL_FLOW_WEIGHT
        )
    elif config.LOSS == 'Huber':
        criterion = HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 添加学习率调度器：验证损失不下降时降低学习率
    # 使用较小的patience（3个epoch）和绝对阈值1e-4，确保在改善不明显时及时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, threshold_mode='abs')

    # 混合精度缩放器（仅当设备为CUDA时启用）——暂时禁用以排查NaN问题
    scaler = None

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10  # 早停耐心值
    
    # 只对流量特征计算损失（因为我们只关心流量预测）
    focus_on_flow = True
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, L, focus_on_flow=focus_on_flow, scaler=scaler)
        val_loss = evaluate(model, val_loader, criterion, device, L, focus_on_flow=focus_on_flow, scaler=scaler)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 如果学习率改变，打印信息
        if old_lr != new_lr:
            print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), args.model_path.replace('.pth', '_best.pth'))
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        if patience_counter >= early_stop_patience:
            print(f"早停触发！验证损失在{early_stop_patience}个epoch内没有改善。")
            break
    
    # 加载最佳模型
    if os.path.exists(args.model_path.replace('.pth', '_best.pth')):
        model.load_state_dict(torch.load(args.model_path.replace('.pth', '_best.pth')))
        print("已加载最佳模型权重。")

    # 保存模型
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"模型已保存至 {args.model_path}")

    # 绘制损失曲线
    plot_curves(train_losses, val_losses, save_path='./results/figures/loss_curve.png')
    return model

def evaluate_model(model_path, data_dir, args):
    """评估模型"""
    print("开始评估...")
    evaluate_on_test(data_dir, model_path,
                     adj_method=args.adj_method,
                     adj_path=args.adj_path,
                     adj_topk=args.adj_topk,
                     adj_threshold=args.adj_threshold,
                     handle_missing=args.handle_missing,
                     handle_outliers_flag=args.handle_outliers,
                     smooth_low_flow_flag=args.smooth_low_flow,
                     outlier_method=args.outlier_method)
    print("评估完成。")

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode in ['preprocess', 'all']:
        X_train, y_train, X_test, y_test, mean, std = preprocess(args.data_dir, args.seq_len, args.pred_len)
    if args.mode in ['train', 'all']:
        if 'X_train' not in locals():
            X_train, y_train, X_test, y_test, mean, std = preprocess(args.data_dir, args.seq_len, args.pred_len)
        train_model(X_train, y_train, X_test, y_test, args)
    if args.mode in ['evaluate', 'all']:
        evaluate_model(args.model_path, args.data_dir, args)

    print("所有任务执行完毕。")

if __name__ == '__main__':
    main()