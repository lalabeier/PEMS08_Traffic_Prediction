# 公路车流量预测项目（课程作业）

基于PEMS08数据集的时空序列预测任务，使用时空图卷积网络（STGCN）预测未来一小时的车流量。

## 项目结构

```
公路车流量预测/
├── data/                    # 数据目录
│   ├── PEMS08_raw/          # 原始数据（需自行放置）
│   └── README.txt           # 数据说明
├── code/                    # 代码目录
│   ├── preprocess.py        # 数据预处理
│   ├── model.py            # STGCN模型定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── utils.py            # 工具函数
├── results/                 # 结果目录
│   ├── models/             # 保存的模型
│   ├── figures/            # 生成的图表
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   └── predictions_10sensors.png
│   └── predictions.csv      # 预测结果
├── config.py               # 配置文件
├── main.py                 # 主程序（一键运行）
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明
```

## 任务要求

- **数据集**：PEMS08数据集，包含170个传感器节点、17856个时间点（小时级）、流量、速度、车道占用率三个特征。
- **目标**：以每小时的流量数据为样本，预测下一个小时的流量。
- **数据划分**：按7:3比例划分为训练集和测试集。
- **评价标准**：预测误差 = (真实流量 - 预测流量) / 真实流量

## 快速开始

### 环境配置

1. 确保已安装Python 3.8+，建议使用虚拟环境。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 数据准备

1. 下载PEMS08数据集（如`pems08.npz`）并放置在 `data/PEMS08_raw/` 目录下。
2. 数据形状应为 `(170, 17856, 3)`。

### 运行流程

#### 一键运行全部流程
```bash
python main.py --mode all
```

#### 分步运行
**启动环境**
   ```bash
   D:/apps/env_py/pytorch_env/Scripts/Activate.ps1
   ```

1. **预处理**：
   ```bash
   D:/apps/env_py/pytorch_env/Scripts/Activate.ps1启动环境
   python main.py --mode preprocess
   ```

2. **训练模型**：
   ```bash
   python main.py --mode train --epochs 50 --batch_size 32
   ```

3. **评估模型**：
   ```bash
   python main.py --mode evaluate
   ```

### 自定义配置

修改 `config.py` 中的参数，或通过命令行参数调整（参见 `main.py --help`）。

## 模型架构

采用时空图卷积网络（STGCN），包括：
- **时间卷积层**：门控TCN捕捉时间依赖。
- **空间卷积层**：切比雪夫图卷积捕捉空间依赖。
- **残差连接**：加速训练并提升性能。

## 结果输出

运行结束后，在 `results/` 目录下将生成：
- `models/stgcn_model.pth`：训练好的模型权重。
- `figures/loss_curve.png`：训练与验证损失曲线。
- `figures/accuracy_curve.png`：准确率曲线（如适用）。
- `figures/predictions_10sensors.png`：前10个传感器的真实值与预测值对比图。
- `predictions.csv`：详细的预测结果表格。


## 注意事项

- 本项目为示例实现，实际效果需根据数据调整。
- 训练过程可能需要GPU加速（配置 `config.py` 中的 `DEVICE`）。
- 原始数据需自行准备，可参考 [PEMS数据集官网](https://pems.dot.ca.gov/)。

## 许可证

MIT License
