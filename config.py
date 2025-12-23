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