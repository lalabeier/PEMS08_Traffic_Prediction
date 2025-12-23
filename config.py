"""
配置文件
"""

# 数据配置
DATA_DIR = './data/PEMS08_raw/'
DATA_SHAPE = (170, 17856, 3)  # 节点数, 时间步数, 特征数
TRAIN_RATIO = 0.7
SEQ_LEN = 12  # 输入序列长度（小时）
PRED_LEN = 1  # 预测长度（小时）

# 模型配置
MODEL_TYPE = 'STGCN'
HIDDEN_CHANNELS = 64
K = 3  # 图卷积阶数
NUM_BLOCKS = 2
DROP_RATE = 0.1

# 训练配置
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = 'Adam'
LOSS = 'MSE'
DEVICE = 'cuda'  # 或 'cpu'

# 路径配置
MODEL_SAVE_DIR = './results/models/'
FIGURE_SAVE_DIR = './results/figures/'
PREDICTION_SAVE_PATH = './results/predictions.csv'

# 日志配置
LOG_DIR = './logs/'
LOG_LEVEL = 'INFO'

# 实验配置
SEED = 42
USE_GPU = True

# 邻接矩阵配置
ADJ_METHOD = 'corr'  # 'corr' 或 'file'
ADJ_PATH = None
ADJ_TOPK = 10
ADJ_THRESHOLD = 0.1

# 数据清洗配置
HANDLE_MISSING = True
HANDLE_OUTLIERS = True
SMOOTH_LOW_FLOW = True
OUTLIER_METHOD = 'cap'  # 'cap', 'smooth', 'remove'
OUTLIER_SPATIOTEMPORAL = True  # 是否使用时序空异常检测
OUTLIER_WINDOW_SIZE = 3  # 时空检测窗口大小
OUTLIER_N_SIGMA = 3.0  # 异常值sigma阈值

# 损失函数配置
LOSS = 'MSE'  # 'MSE', 'WeightedMSE', 'Huber'
USE_WEIGHTED_LOSS = True  # 是否使用加权损失
LOW_FLOW_THRESHOLD = 100.0  # 低流量阈值
HIGH_FLOW_THRESHOLD = 500.0  # 高流量阈值
LOW_FLOW_WEIGHT = 2.0  # 低流量样本权重
HIGH_FLOW_WEIGHT = 1.5  # 高流量样本权重
NORMAL_FLOW_WEIGHT = 1.0  # 正常流量样本权重

# 模型增强配置
USE_ATTENTION = True  # 是否在STGCN中添加注意力模块
ATTENTION_HEADS = 4  # 注意力头数
ATTENTION_DROPOUT = 0.1  # 注意力dropout率

# 评估配置
ERROR_BINS = [0, 100, 300, 500, 1000]  # 误差分区间隔（流量值）

# 模型路径配置
MODEL_PATH = './results/models/stgcn_model.pth'