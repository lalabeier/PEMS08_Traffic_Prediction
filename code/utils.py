"""
工具函数
"""

import numpy as np
import torch
import os
import json

def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_config(config, path='../results/config.json'):
    """保存配置字典为JSON"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    """加载JSON配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(y_true, y_pred):
    """计算评估指标：MAE, RMSE, MAPE"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # 避免除以零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def generate_adjacency_matrix(num_nodes, connection_radius=0.1):
    """生成一个随机的邻接矩阵（用于示例）"""
    adj = np.random.rand(num_nodes, num_nodes)
    adj = adj * (adj < connection_radius)
    adj = adj + adj.T  # 对称
    adj = (adj > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return adj

def load_adjacency_from_file(path, num_nodes=None):
    """
    从文件加载邻接矩阵，支持 .npy 或 .csv
    Args:
        path: 文件路径
        num_nodes: 可选，验证节点数
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"邻接矩阵文件不存在: {path}")
    if path.endswith('.npy'):
        adj = np.load(path)
    else:
        adj = np.loadtxt(path, delimiter=',')
    if num_nodes is not None and adj.shape[0] != num_nodes:
        raise ValueError(f"邻接矩阵节点数不匹配，期望 {num_nodes}，实际 {adj.shape[0]}")
    return adj

def build_corr_adjacency(data, k=10, threshold=0.1, use_abs=True):
    """
    基于历史流量的相关性构建邻接矩阵
    Args:
        data: (nodes, timesteps, features)，第0个特征视为流量
        k: 每个节点保留的最大邻居数（按相关性排序）
        threshold: 相关性阈值，低于该值视为0
        use_abs: 是否使用绝对值相关性
    Returns:
        adj: (num_nodes, num_nodes) numpy array
    """
    flow = data[:, :, 0]  # (nodes, timesteps)
    corr = np.corrcoef(flow)  # (nodes, nodes)
    if use_abs:
        corr = np.abs(corr)
    np.fill_diagonal(corr, 0)

    # 阈值过滤
    corr[corr < threshold] = 0

    # 每个节点保留 top-k
    if k is not None and k > 0:
        for i in range(corr.shape[0]):
            row = corr[i]
            if np.count_nonzero(row) > k:
                # 找到第k大值作为截断
                kth = np.partition(row, -k)[-k]
                row[row < kth] = 0
                corr[i] = row

    # 对称化
    adj = np.maximum(corr, corr.T)
    return adj

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_model_summary(model, input_shape):
    """打印模型概要"""
    from torchsummary import summary
    summary(model, input_shape)

if __name__ == '__main__':
    # 示例用法
    set_seed()
    adj = generate_adjacency_matrix(10)
    print(adj.shape)
    print(compute_metrics(np.array([1,2,3]), np.array([1.1,1.9,3.2])))