"""
评估脚本
计算预测误差并生成图表。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ERROR_BINS

from model import STGCN, get_laplacian
from preprocess import load_raw_data, split_train_test, normalize, create_sequences
from utils import build_corr_adjacency, load_adjacency_from_file

def load_model(model_path, num_nodes, device='cpu', in_channels=3, out_channels=3):
    """加载训练好的模型"""
    model = STGCN(num_nodes, in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_prediction_error(y_true, y_pred):
    """计算预测误差：(真实流量 - 预测流量) / 真实流量
    对于真实值很小的情况，使用改进的处理方式避免除以0
    """
    # 使用一个阈值，避免除以非常小的数
    threshold = 1.0
    mask_valid = np.abs(y_true) > threshold
    
    error = np.zeros_like(y_true)
    
    # 对于真实值足够大的样本，使用相对误差
    if np.any(mask_valid):
        error[mask_valid] = (y_true[mask_valid] - y_pred[mask_valid]) / y_true[mask_valid]
    
    # 对于真实值很小的样本，使用归一化的绝对误差（除以平均值而不是真实值）
    if np.any(~mask_valid):
        mean_true = np.mean(np.abs(y_true[y_true != 0])) if np.any(y_true != 0) else 1.0
        if mean_true > 1e-6:
            error[~mask_valid] = (y_true[~mask_valid] - y_pred[~mask_valid]) / mean_true
        else:
            error[~mask_valid] = 0  # 如果所有真实值都为0，误差设为0
    
    return error

def compute_binned_errors(y_true, y_pred, bins):
    """
    根据真实流量值分区间计算误差统计量。
    
    参数：
        y_true, y_pred: 形状 (samples, nodes, timesteps) 或展平后的1D数组
        bins: 区间边界列表，例如 [0, 100, 300, 500, 1000]
        
    返回：
        results: 列表，每个元素为一个字典，包含区间统计信息
    """
    # 展平为一维
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 确保bins是单调递增的
    bins = np.array(bins)
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bins must be strictly increasing")
    
    # 分配样本到区间 (左闭右开区间)
    # 使用 np.digitize，返回区间索引（1到len(bins)）
    indices = np.digitize(y_true_flat, bins)
    # 注意：np.digitize 返回 i 使得 bins[i-1] <= x < bins[i]（如果bins递增）
    # 对于 x < bins[0] 返回0，对于 x >= bins[-1] 返回 len(bins)
    # 我们将处理边界情况：将索引0合并到第一个区间，将索引len(bins)合并到最后一个区间
    # 调整区间索引
    n_bins = len(bins) - 1  # 区间数量
    results = []
    for i in range(n_bins):
        # 区间 i: bins[i] <= y_true < bins[i+1]
        mask = (indices == i+1)  # digitize 返回 i+1
        # 包括小于 bins[0] 的样本（索引0）到第一个区间吗？通常 bins[0] 是最小值，我们可以忽略或合并
        # 这里我们将忽略小于 bins[0] 的样本（因为 bins[0] 通常为0）
        # 对于第一个区间，也包含索引0的样本（即 y_true < bins[0]）
        if i == 0:
            mask = mask | (indices == 0)
        
        y_true_bin = y_true_flat[mask]
        y_pred_bin = y_pred_flat[mask]
        count = len(y_true_bin)
        
        if count == 0:
            results.append({
                'bin': f"[{bins[i]}, {bins[i+1]})",
                'count': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'mean_true': np.nan,
                'mean_pred': np.nan
            })
            continue
        
        # 计算指标
        rmse = np.sqrt(np.mean((y_true_bin - y_pred_bin) ** 2))
        mae = np.mean(np.abs(y_true_bin - y_pred_bin))
        # MAPE（避免除以0）
        mask_nonzero = y_true_bin != 0
        if np.any(mask_nonzero):
            mape = np.mean(np.abs((y_true_bin[mask_nonzero] - y_pred_bin[mask_nonzero]) / y_true_bin[mask_nonzero])) * 100
        else:
            mape = np.nan
        # R²
        ss_res = np.sum((y_true_bin - y_pred_bin) ** 2)
        ss_tot = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        
        results.append({
            'bin': f"[{bins[i]}, {bins[i+1]})",
            'count': count,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'mean_true': np.mean(y_true_bin),
            'mean_pred': np.mean(y_pred_bin)
        })
    
    # 处理最后一个区间外的样本（y_true >= bins[-1]）
    mask_last = (indices == len(bins))
    if np.any(mask_last):
        y_true_bin = y_true_flat[mask_last]
        y_pred_bin = y_pred_flat[mask_last]
        count = len(y_true_bin)
        rmse = np.sqrt(np.mean((y_true_bin - y_pred_bin) ** 2))
        mae = np.mean(np.abs(y_true_bin - y_pred_bin))
        mask_nonzero = y_true_bin != 0
        if np.any(mask_nonzero):
            mape = np.mean(np.abs((y_true_bin[mask_nonzero] - y_pred_bin[mask_nonzero]) / y_true_bin[mask_nonzero])) * 100
        else:
            mape = np.nan
        ss_res = np.sum((y_true_bin - y_pred_bin) ** 2)
        ss_tot = np.sum((y_true_bin - np.mean(y_true_bin)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        results.append({
            'bin': f"[{bins[-1]}, inf)",
            'count': count,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'mean_true': np.mean(y_true_bin),
            'mean_pred': np.mean(y_pred_bin)
        })
    
    return results

def evaluate_on_test(data_dir='../data/PEMS08_raw/', model_path='../results/models/stgcn_model.pth',
                     adj_method='corr', adj_path=None, adj_topk=10, adj_threshold=0.1,
                     handle_missing=True, handle_outliers_flag=True, smooth_low_flow_flag=True,
                     outlier_method='cap', add_time_features=True, time_features_for_input_only=True):
    """在测试集上评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 12
    pred_len = 1

    # 处理路径：如果是相对路径，转换为相对于项目根目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # code目录的父目录，即项目根目录
    
    if not os.path.isabs(data_dir):
        # 处理相对路径：移除../或./前缀，然后与项目根目录拼接
        data_dir_clean = data_dir.replace('../', '').replace('./', '')
        data_dir = os.path.normpath(os.path.join(project_root, data_dir_clean))
    if not os.path.isabs(model_path):
        model_path_clean = model_path.replace('../', '').replace('./', '')
        model_path = os.path.normpath(os.path.join(project_root, model_path_clean))

    # 加载数据
    data = load_raw_data(data_dir,
                         handle_missing=handle_missing,
                         handle_outliers_flag=handle_outliers_flag,
                         smooth_low_flow_flag=smooth_low_flow_flag,
                         outlier_method=outlier_method,
                         verbose=True)
    train_data, test_data = split_train_test(data)
    train_norm, mean, std = normalize(train_data)
    test_norm, _, _ = normalize(test_data, mean, std)
    X_test, y_test = create_sequences(test_norm, seq_len, pred_len,
                                      add_time_features_flag=add_time_features,
                                      time_features_for_input_only=time_features_for_input_only)

    # 确定输入输出通道数
    in_channels = X_test.shape[3]  # 特征维度
    out_channels = y_test.shape[3]
    print(f"输入通道数: {in_channels}, 输出通道数: {out_channels}")
    
    # 转换为torch张量
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)  # (samples, features, nodes, seq_len)
    y_test = torch.FloatTensor(y_test).permute(0, 3, 1, 2)

    num_nodes = X_test.shape[2]
    if adj_method == 'file':
        if not adj_path:
            raise ValueError("adj_method=file 时必须提供 adj_path")
        adj_np = load_adjacency_from_file(adj_path, num_nodes=num_nodes)
    else:
        # 基于相关性构建邻接矩阵（使用清洗后的全量数据）
        adj_np = build_corr_adjacency(data, k=adj_topk, threshold=adj_threshold, use_abs=True)
    adj = torch.FloatTensor(adj_np)
    L = get_laplacian(adj).to(device)

    # 加载模型
    model = load_model(model_path, num_nodes, device, in_channels=in_channels, out_channels=out_channels)

    # 预测
    predictions = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size].to(device)
            batch_pred = model(batch_X, L)
            predictions.append(batch_pred.cpu())
    y_pred = torch.cat(predictions, dim=0)

    # 反标准化（mean和std的形状是(1, 1, features)，需要reshape为(1, features, 1, 1)以匹配(samples, features, nodes, timesteps)）
    # mean和std的最后一个维度是features维度，需要将其reshape到第二个维度
    mean_reshaped = mean.squeeze()  # (features,) -> 去掉前两个维度
    std_reshaped = std.squeeze()    # (features,)
    # 确保特征数匹配输出通道数
    if mean_reshaped.shape[0] != out_channels:
        # 如果特征数不匹配，可能时间特征被添加到了输入但未包含在输出中，我们只取前out_channels个特征
        mean_reshaped = mean_reshaped[:out_channels]
        std_reshaped = std_reshaped[:out_channels]
    mean_tensor = torch.FloatTensor(mean_reshaped).reshape(1, -1, 1, 1)  # (1, out_channels, 1, 1)
    std_tensor = torch.FloatTensor(std_reshaped).reshape(1, -1, 1, 1)    # (1, out_channels, 1, 1)
    y_true_orig = y_test * std_tensor + mean_tensor
    y_pred_orig = y_pred * std_tensor + mean_tensor

    # 提取流量特征（第一个通道，索引0）进行误差计算
    y_true_flow = y_true_orig[:, 0, :, :].numpy()  # (samples, nodes, timesteps)
    y_pred_flow = y_pred_orig[:, 0, :, :].numpy()  # (samples, nodes, timesteps)

    # 计算误差（只计算真实值不为0的样本）
    # 为了避免除以0导致的极大误差，只对真实值>阈值的样本计算相对误差
    threshold = 1.0  # 真实值小于此阈值时，使用绝对误差而不是相对误差
    mask_valid = np.abs(y_true_flow) > threshold
    
    if np.any(mask_valid):
        # 对于真实值足够大的样本，使用相对误差
        error_valid = (y_true_flow[mask_valid] - y_pred_flow[mask_valid]) / y_true_flow[mask_valid]
        mean_error = np.mean(np.abs(error_valid))
        print(f"平均绝对预测误差（相对误差，真实值>{threshold}）：{mean_error:.4f}")
        print(f"有效样本数：{np.sum(mask_valid)} / {y_true_flow.size} ({100*np.sum(mask_valid)/y_true_flow.size:.2f}%)")
        
        # 计算MAPE（Mean Absolute Percentage Error）
        mape = np.mean(np.abs(error_valid)) * 100
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        
        # 计算RMSE（均方根误差）
        rmse = np.sqrt(np.mean((y_true_flow[mask_valid] - y_pred_flow[mask_valid]) ** 2))
        print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        
        # 计算R²（决定系数）
        y_true_valid = y_true_flow[mask_valid]
        y_pred_valid = y_pred_flow[mask_valid]
        ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
        ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        print(f"R^2 (Coefficient of Determination): {r2:.4f}")
        
        # 对于真实值很小的样本，计算绝对误差
        if np.any(~mask_valid):
            mae_small = np.mean(np.abs(y_true_flow[~mask_valid] - y_pred_flow[~mask_valid]))
            print(f"真实值<={threshold}的样本数量：{np.sum(~mask_valid)}，平均绝对误差：{mae_small:.4f}")
    else:
        # 如果所有真实值都很小，使用绝对误差
        mean_error = np.mean(np.abs(y_true_flow - y_pred_flow))
        print(f"平均绝对预测误差（绝对误差）：{mean_error:.4f}")
        # 计算整体RMSE和R²
        rmse = np.sqrt(np.mean((y_true_flow - y_pred_flow) ** 2))
        ss_res = np.sum((y_true_flow - y_pred_flow) ** 2)
        ss_tot = np.sum((y_true_flow - np.mean(y_true_flow)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        print(f"R^2 (Coefficient of Determination): {r2:.4f}")
    
    # 为了兼容后续代码，仍然计算全量误差（但会包含异常值）
    error = compute_prediction_error(y_true_flow, y_pred_flow)

    # 分区间误差统计
    binned_results = compute_binned_errors(y_true_flow, y_pred_flow, ERROR_BINS)
    print("\n=== 分区间误差统计 ===")
    for res in binned_results:
        if res['count'] > 0:
            print(f"区间 {res['bin']}: 样本数 {res['count']}, RMSE {res['rmse']:.2f}, MAE {res['mae']:.2f}, MAPE {res['mape']:.2f}%, R² {res['r2']:.4f}")
            print(f"    平均真实值 {res['mean_true']:.2f}, 平均预测值 {res['mean_pred']:.2f}")
        else:
            print(f"区间 {res['bin']}: 无样本")
    print("=====================\n")

    # 保存预测结果（使用项目根目录的路径）
    results_dir = os.path.join(project_root, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, 'predictions.npy'), y_pred_orig.numpy())
    np.save(os.path.join(results_dir, 'ground_truth.npy'), y_true_orig.numpy())
    # 保存为CSV（示例：前10个节点）
    df = pd.DataFrame({
        'node': np.repeat(range(10), y_true_orig.shape[0]),
        'timestep': np.tile(range(y_true_orig.shape[0]), 10),
        'true': y_true_orig.numpy()[:, 0, :10, 0].flatten(),
        'pred': y_pred_orig.numpy()[:, 0, :10, 0].flatten()
    })
    df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    print(f"预测结果已保存至 {results_dir}")

    # 绘制10个传感器的对比图
    plot_path = os.path.join(figures_dir, 'predictions_10sensors.png')
    plot_10_sensors(y_true_orig.numpy(), y_pred_orig.numpy(), save_path=plot_path)
    
    # 绘制准确率曲线（基于每个时间步的预测准确率）
    accuracy_curve_path = os.path.join(figures_dir, 'accuracy_curve.png')
    plot_accuracy_curve(y_true_flow, y_pred_flow, save_path=accuracy_curve_path)

    return error

def plot_10_sensors(y_true, y_pred, save_path):
    """绘制10个传感器的真实流量与预测流量对比图"""
    n_sensors = 10
    timesteps = y_true.shape[0]
    plt.figure(figsize=(15, 10))
    for i in range(n_sensors):
        plt.subplot(5, 2, i+1)
        plt.plot(range(timesteps), y_true[:, 0, i, 0], label='True', alpha=0.7)
        plt.plot(range(timesteps), y_pred[:, 0, i, 0], label='Pred', alpha=0.7)
        plt.title(f'Sensor {i}')
        plt.xlabel('Time step')
        plt.ylabel('Flow')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"对比图已保存至 {save_path}")

def plot_accuracy_curve(y_true, y_pred, save_path):
    """绘制准确率曲线
    准确率定义为：max(0, 1 - |相对误差|)，其中相对误差 = (真实值 - 预测值) / 真实值
    或者使用 MAPE (Mean Absolute Percentage Error) 的逆：accuracy = 1 / (1 + MAPE)
    """
    # y_true和y_pred的形状：(samples, nodes, timesteps)
    # 计算相对误差的绝对值
    abs_error = np.abs((y_true - y_pred) / (y_true + 1e-8))  # 相对误差的绝对值
    
    # 方法1：准确率 = max(0, 1 - |相对误差|)，限制在[0, 1]
    accuracy = np.maximum(0, 1 - abs_error)
    
    # 对每个样本，计算所有节点的平均准确率
    # 如果y_true形状是(samples, nodes, timesteps)，axis=1是对nodes维度求平均
    sample_accuracy = np.mean(accuracy, axis=1)  # (samples, timesteps)
    
    # 如果是2D（samples, timesteps），取第一个时间步（因为pred_len=1）
    if len(sample_accuracy.shape) == 2:
        sample_accuracy = sample_accuracy[:, 0]  # (samples,)
    
    # 绘制曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sample_accuracy)), sample_accuracy * 100, alpha=0.7, linewidth=1, label='Accuracy')
    plt.xlabel('Sample Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Prediction Accuracy Curve (Accuracy = max(0, 1 - |Relative Error|))')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    
    # 添加平均准确率标注
    mean_acc = np.mean(sample_accuracy) * 100
    plt.axhline(y=mean_acc, color='r', linestyle='--', alpha=0.7, label=f'Mean Accuracy: {mean_acc:.2f}%')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"准确率曲线已保存至 {save_path}")
    print(f"平均准确率：{mean_acc:.2f}%")

if __name__ == '__main__':
    evaluate_on_test()