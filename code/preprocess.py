"""
数据预处理脚本
功能：加载PEMS08数据集，划分训练集和测试集，进行标准化等处理。
增加了异常值检测、缺失值处理和低流量平滑。
"""

import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.ndimage import uniform_filter1d

def detect_and_handle_missing(data, verbose=True):
    """
    检测和处理缺失值（NaN或Inf）
    Args:
        data: (nodes, timesteps, features)
    Returns:
        cleaned_data: 处理后的数据
        stats: 统计信息字典
    """
    original_shape = data.shape
    stats_dict = {
        'missing_count': np.sum(~np.isfinite(data)),
        'missing_percentage': 0.0,
        'missing_per_feature': []
    }
    
    if stats_dict['missing_count'] > 0:
        stats_dict['missing_percentage'] = stats_dict['missing_count'] / data.size * 100
        if verbose:
            print(f"检测到缺失值：{stats_dict['missing_count']} 个 ({stats_dict['missing_percentage']:.2f}%)")
        
        # 对每个特征分别处理
        cleaned_data = np.copy(data)
        for f in range(data.shape[2]):
            feature_data = data[:, :, f]
            missing_count = np.sum(~np.isfinite(feature_data))
            stats_dict['missing_per_feature'].append(missing_count)
            
            if missing_count > 0:
                # 使用前向填充，如果第一个值缺失则用后向填充
                for node_idx in range(feature_data.shape[0]):
                    node_series = feature_data[node_idx, :]
                    valid_mask = np.isfinite(node_series)
                    
                    if not np.all(valid_mask):
                        # 前向填充
                        node_series = pd.Series(node_series).fillna(method='ffill')
                        # 后向填充（处理开头缺失）
                        node_series = pd.Series(node_series).fillna(method='bfill')
                        # 如果还有缺失（全为NaN），用0填充
                        node_series = node_series.fillna(0)
                        cleaned_data[node_idx, :, f] = node_series.values
        
        if verbose:
            print(f"缺失值已处理：前向/后向填充")
    else:
        cleaned_data = data
        if verbose:
            print("未检测到缺失值")
    
    return cleaned_data, stats_dict

def detect_outliers(data, z_threshold=3.0, iqr_factor=1.5, verbose=True):
    """
    检测异常值
    Args:
        data: (nodes, timesteps, features)
        z_threshold: Z-score阈值
        iqr_factor: IQR方法的倍数因子
    Returns:
        outlier_mask: 异常值掩码，True表示异常值
        stats: 统计信息字典
    """
    stats_dict = {
        'outlier_count': 0,
        'outlier_percentage': 0.0,
        'outlier_per_feature': []
    }
    
    outlier_mask = np.zeros_like(data, dtype=bool)
    
    for f in range(data.shape[2]):
        feature_data = data[:, :, f].flatten()
        
        # 方法1: Z-score方法
        z_scores = np.abs(stats.zscore(feature_data, nan_policy='omit'))
        z_outliers = z_scores > z_threshold
        
        # 方法2: IQR方法
        Q1 = np.percentile(feature_data, 25)
        Q3 = np.percentile(feature_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        iqr_outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
        
        # 合并两种方法的检测结果
        feature_outliers = z_outliers | iqr_outliers
        outlier_count = np.sum(feature_outliers)
        stats_dict['outlier_per_feature'].append(outlier_count)
        
        # 恢复形状并标记
        feature_mask = feature_outliers.reshape(data.shape[:2])
        outlier_mask[:, :, f] = feature_mask
    
    stats_dict['outlier_count'] = np.sum(outlier_mask)
    stats_dict['outlier_percentage'] = stats_dict['outlier_count'] / data.size * 100
    
    if verbose:
        print(f"检测到异常值：{stats_dict['outlier_count']} 个 ({stats_dict['outlier_percentage']:.2f}%)")
        for f in range(data.shape[2]):
            print(f"  特征 {f}: {stats_dict['outlier_per_feature'][f]} 个异常值")
    
    return outlier_mask, stats_dict

def detect_spatiotemporal_outliers(data, window_size=3, n_sigma=3.0, verbose=True):
    """
    基于时空邻域的异常值检测
    Args:
        data: (nodes, timesteps, features)
        window_size: 时间维度滑动窗口大小（奇数）
        n_sigma: 标准差倍数阈值
        verbose: 是否打印详细信息
    Returns:
        outlier_mask: 异常值掩码，True表示异常值
        stats: 统计信息字典
    """
    from scipy.ndimage import uniform_filter
    
    stats_dict = {
        'outlier_count': 0,
        'outlier_percentage': 0.0,
        'outlier_per_feature': []
    }
    
    outlier_mask = np.zeros_like(data, dtype=bool)
    
    for f in range(data.shape[2]):
        feature_data = data[:, :, f]
        
        # 计算局部均值和标准差（仅时间维度滑动窗口）
        # uniform_filter 用于计算滑动平均
        local_mean = uniform_filter(feature_data, size=(1, window_size), mode='nearest')
        local_std = np.sqrt(uniform_filter((feature_data - local_mean)**2,
                                          size=(1, window_size), mode='nearest'))
        
        # 避免除零
        local_std[local_std < 1e-8] = 1e-8
        
        # 计算局部Z-score
        z_scores = np.abs((feature_data - local_mean) / local_std)
        feature_mask = z_scores > n_sigma
        
        outlier_mask[:, :, f] = feature_mask
        outlier_count = np.sum(feature_mask)
        stats_dict['outlier_per_feature'].append(outlier_count)
    
    stats_dict['outlier_count'] = np.sum(outlier_mask)
    stats_dict['outlier_percentage'] = stats_dict['outlier_count'] / data.size * 100
    
    if verbose:
        print(f"时空异常值检测：{stats_dict['outlier_count']} 个 ({stats_dict['outlier_percentage']:.2f}%)")
        for f in range(data.shape[2]):
            print(f"  特征 {f}: {stats_dict['outlier_per_feature'][f]} 个异常值")
    
    return outlier_mask, stats_dict


def handle_outliers(data, outlier_mask, method='cap', verbose=True):
    """
    处理异常值
    Args:
        data: (nodes, timesteps, features)
        outlier_mask: 异常值掩码
        method: 'cap'（截断）或 'smooth'（平滑）或 'remove'（移除为NaN，后续填充）
    Returns:
        cleaned_data: 处理后的数据
    """
    cleaned_data = np.copy(data)
    
    if np.sum(outlier_mask) == 0:
        if verbose:
            print("无需处理异常值")
        return cleaned_data
    
    for f in range(data.shape[2]):
        feature_data = cleaned_data[:, :, f]
        feature_mask = outlier_mask[:, :, f]
        
        if np.sum(feature_mask) > 0:
            if method == 'cap':
                # 截断到合理范围（使用1%和99%分位数）
                p1, p99 = np.percentile(feature_data[~feature_mask], [1, 99])
                feature_data[feature_data < p1] = p1
                feature_data[feature_data > p99] = p99
                if verbose and f == 0:  # 只打印一次
                    print(f"异常值处理方法: 截断到 [{p1:.2f}, {p99:.2f}]")
            elif method == 'smooth':
                # 使用时间平滑（只对时间维度平滑）
                for node_idx in range(feature_data.shape[0]):
                    node_series = feature_data[node_idx, :]
                    if np.sum(feature_mask[node_idx, :]) > 0:
                        # 使用滑动窗口平滑
                        smoothed = uniform_filter1d(node_series, size=3, mode='nearest')
                        node_series[feature_mask[node_idx, :]] = smoothed[feature_mask[node_idx, :]]
                        feature_data[node_idx, :] = node_series
                if verbose and f == 0:
                    print("异常值处理方法: 时间平滑")
            elif method == 'remove':
                # 标记为NaN，后续会被缺失值处理填充
                feature_data[feature_mask] = np.nan
                if verbose and f == 0:
                    print("异常值处理方法: 移除（标记为NaN）")
            
            cleaned_data[:, :, f] = feature_data
    
    return cleaned_data

def smooth_low_flow(data, flow_threshold=1.0, window_size=3, verbose=True):
    """
    对低流量段进行平滑处理
    Args:
        data: (nodes, timesteps, features)，假设第一个特征是流量
        flow_threshold: 低流量的阈值
        window_size: 平滑窗口大小
    Returns:
        smoothed_data: 平滑后的数据
        stats: 统计信息
    """
    smoothed_data = np.copy(data)
    flow_data = data[:, :, 0]  # 假设第一个特征是流量
    
    # 找出低流量位置
    low_flow_mask = flow_data < flow_threshold
    low_flow_count = np.sum(low_flow_mask)
    
    if low_flow_count > 0:
        if verbose:
            print(f"检测到低流量段（< {flow_threshold}）: {low_flow_count} 个数据点 ({100*low_flow_count/flow_data.size:.2f}%)")
        
        # 对每个节点的低流量段进行平滑
        for node_idx in range(flow_data.shape[0]):
            node_flow = flow_data[node_idx, :]
            node_mask = low_flow_mask[node_idx, :]
            
            if np.sum(node_mask) > 0:
                # 对流量特征使用滑动窗口平滑
                smoothed_flow = uniform_filter1d(node_flow, size=window_size, mode='nearest')
                # 只更新低流量段
                smoothed_data[node_idx, node_mask, 0] = smoothed_flow[node_mask]
        
        if verbose:
            print(f"低流量段已平滑（窗口大小: {window_size}）")
    else:
        if verbose:
            print("未检测到需要平滑的低流量段")
    
    stats = {
        'low_flow_count': low_flow_count,
        'low_flow_percentage': 100 * low_flow_count / flow_data.size
    }
    
    return smoothed_data, stats

def load_raw_data(data_dir='../data/PEMS08_raw/', handle_missing=True, handle_outliers_flag=True,
                  smooth_low_flow_flag=True, outlier_method='cap', verbose=True, use_spatiotemporal=False):
    """
    加载原始数据并进行数据清洗。
    假设原始数据为.npz文件，包含'data'键，形状为(170, 17856, 3)。
    如果实际形状是(timesteps, nodes, features)，则转置为(nodes, timesteps, features)。
    
    Args:
        handle_missing: 是否处理缺失值
        handle_outliers_flag: 是否处理异常值
        smooth_low_flow_flag: 是否平滑低流量段
        outlier_method: 异常值处理方法 ('cap', 'smooth', 'remove')
        verbose: 是否打印详细信息
        use_spatiotemporal: 是否使用时空异常值检测（否则使用传统检测）
    """
    data_path = os.path.join(data_dir, 'pems08.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"原始数据文件不存在：{data_path}")
    data = np.load(data_path)['data']
    if verbose:
        print(f"加载数据形状：{data.shape}")
    
    # 确保形状为(nodes, timesteps, features)
    if data.shape[0] == 17856 and data.shape[1] == 170:
        # 原始数据可能是(timesteps, nodes, features)，转置
        data = data.transpose(1, 0, 2)
        if verbose:
            print(f"转置后形状：{data.shape}")
    
    # 数据清洗流程
    if verbose:
        print("\n=== 开始数据清洗 ===")
    
    # 1. 处理缺失值
    if handle_missing:
        data, missing_stats = detect_and_handle_missing(data, verbose=verbose)
    
    # 2. 处理异常值
    if handle_outliers_flag:
        if use_spatiotemporal:
            outlier_mask, outlier_stats = detect_spatiotemporal_outliers(
                data, window_size=3, n_sigma=3.0, verbose=verbose
            )
        else:
            outlier_mask, outlier_stats = detect_outliers(data, verbose=verbose)
        data = handle_outliers(data, outlier_mask, method=outlier_method, verbose=verbose)
        # 如果使用remove方法，需要再次处理缺失值
        if outlier_method == 'remove' and handle_missing:
            data, _ = detect_and_handle_missing(data, verbose=False)
    
    # 3. 平滑低流量段
    if smooth_low_flow_flag:
        data, smooth_stats = smooth_low_flow(data, verbose=verbose)
    
    if verbose:
        print("=== 数据清洗完成 ===\n")
    
    return data

def split_train_test(data, train_ratio=0.7):
    """
    按时间维度划分训练集和测试集。
    数据形状：(nodes, timesteps, features)
    沿timesteps划分。
    """
    nodes, timesteps, features = data.shape
    split_idx = int(timesteps * train_ratio)
    train_data = data[:, :split_idx, :]
    test_data = data[:, split_idx:, :]
    print(f"训练集形状：{train_data.shape}，测试集形状：{test_data.shape}")
    return train_data, test_data

def normalize(data, mean=None, std=None):
    """
    Z-score标准化。
    如果未提供mean和std，则计算数据的均值和标准差。
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=(0,1), keepdims=True)
        std = np.std(data, axis=(0,1), keepdims=True)
    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std

def add_time_features(data):
    """
    向数据添加时间特征（小时、星期几）。
    假设时间步长为1小时，起始时间为周一00:00。
    输入数据形状：(nodes, timesteps, features)
    返回数据形状：(nodes, timesteps, features + 2)
    """
    nodes, timesteps, features = data.shape
    # 创建时间索引（0到timesteps-1）
    time_idx = np.arange(timesteps)
    hour = (time_idx % 24) / 23.0  # 归一化到[0,1]
    day_of_week = ((time_idx // 24) % 7) / 6.0  # 归一化到[0,1]
    # 广播到所有节点
    hour_full = np.tile(hour, (nodes, 1)).reshape(nodes, timesteps, 1)
    day_full = np.tile(day_of_week, (nodes, 1)).reshape(nodes, timesteps, 1)
    # 拼接特征
    data_with_time = np.concatenate([data, hour_full, day_full], axis=-1)
    return data_with_time

def create_sequences(data, seq_len=12, pred_len=1, add_time_features_flag=True, time_features_for_input_only=True):
    """
    创建滑动窗口序列用于时序预测。
    输入数据形状：(nodes, timesteps, features)
    返回X, y形状：(samples, nodes, seq_len, features) 和 (samples, nodes, pred_len, features)
    如果add_time_features_flag为True，则自动添加时间特征（小时、星期几）作为额外特征。
    如果time_features_for_input_only为True，则时间特征仅添加到输入X，而不添加到输出y。
    此时，X的特征数 = 原始特征数 + 2，y的特征数 = 原始特征数。
    """
    original_features = data.shape[2]
    if add_time_features_flag:
        data_with_time = add_time_features(data)  # 形状 (nodes, timesteps, original_features + 2)
        if time_features_for_input_only:
            # 对于输入，使用带时间特征的数据；对于输出，使用原始数据（但需要对齐时间步）
            # 我们需要确保X和y的时间步对齐
            nodes, timesteps, features_with_time = data_with_time.shape
            X, y = [], []
            for i in range(timesteps - seq_len - pred_len + 1):
                X.append(data_with_time[:, i:i+seq_len, :])
                y.append(data[:, i+seq_len:i+seq_len+pred_len, :])
            X = np.array(X)  # (samples, nodes, seq_len, features_with_time)
            y = np.array(y)  # (samples, nodes, pred_len, original_features)
            return X, y
        else:
            data = data_with_time
    # 默认情况（不添加时间特征，或时间特征同时添加到X和y）
    nodes, timesteps, features = data.shape
    X, y = [], []
    for i in range(timesteps - seq_len - pred_len + 1):
        X.append(data[:, i:i+seq_len, :])
        y.append(data[:, i+seq_len:i+seq_len+pred_len, :])
    X = np.array(X)  # (samples, nodes, seq_len, features)
    y = np.array(y)  # (samples, nodes, pred_len, features)
    return X, y

if __name__ == '__main__':
    # 示例用法
    data = load_raw_data()
    train, test = split_train_test(data)
    train_norm, mean, std = normalize(train)
    test_norm, _, _ = normalize(test, mean, std)
    X_train, y_train = create_sequences(train_norm)
    X_test, y_test = create_sequences(test_norm)
    print("预处理完成。")