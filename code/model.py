"""
时空图卷积网络（STGCN）模型定义
参考：https://arxiv.org/abs/1709.04875
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """时间卷积层（门控TCN）"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding))
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: (batch, channels, nodes, timesteps)
        """
        gate = torch.sigmoid(self.conv2(x))
        out = self.conv1(x) * gate
        out = self.batch_norm(out)
        return F.relu(out)

class SpatialConv(nn.Module):
    """空间图卷积层（ChebConv）"""
    def __init__(self, in_channels, out_channels, K=3):
        super(SpatialConv, self).__init__()
        self.K = K
        self.weights = nn.Parameter(torch.randn(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, L):
        """
        x: (batch, channels, nodes, timesteps)
        L: 拉普拉斯矩阵 (nodes, nodes)
        """
        batch, channels, nodes, timesteps = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, nodes, channels, timesteps)
        x = x.view(batch * nodes, channels, timesteps)
        # 正确的图卷积（对k求和，输出out_channels）
        out = torch.einsum('kio,bct->bot', self.weights, x)
        out = out.view(batch, nodes, -1, timesteps)  # -1 应为 out_channels
        out = out.permute(0, 2, 1, 3)  # (batch, out_channels, nodes, timesteps)
        out = out + self.bias.view(1, -1, 1, 1)
        return F.relu(out)

class STGCNBlock(nn.Module):
    """STGCN块：时间卷积 + 空间卷积 + 时间卷积"""
    def __init__(self, in_channels, out_channels, K=3):
        super(STGCNBlock, self).__init__()
        self.temp1 = TemporalConv(in_channels, out_channels)
        self.spatial = SpatialConv(out_channels, out_channels, K)
        self.temp2 = TemporalConv(out_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, L):
        residual = self.residual(x)
        out = self.temp1(x)
        out = self.spatial(out, L)
        out = self.temp2(out)
        out = out + residual
        return out

class STGCN(nn.Module):
    """完整的STGCN模型"""
    def __init__(self, num_nodes, in_channels=3, hidden_channels=64, out_channels=1, K=3, num_blocks=2):
        super(STGCN, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(STGCNBlock(in_channels, hidden_channels, K))
        for _ in range(num_blocks - 1):
            self.blocks.append(STGCNBlock(hidden_channels, hidden_channels, K))
        self.final_temp = TemporalConv(hidden_channels, hidden_channels)
        self.output = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x, L):
        """
        x: (batch, channels, nodes, timesteps)
        L: 拉普拉斯矩阵 (nodes, nodes)
        """
        for block in self.blocks:
            x = block(x, L)
        x = self.final_temp(x)
        x = self.output(x)
        # 取最后一个时间步作为预测（假设 pred_len = 1）
        x = x[:, :, :, -1:]
        return x  # (batch, out_channels, nodes, 1)

def get_laplacian(adj):
    """计算归一化拉普拉斯矩阵"""
    D = torch.diag(adj.sum(dim=1))
    L = D - adj
    return L

if __name__ == '__main__':
    # 示例：创建模型
    num_nodes = 170
    model = STGCN(num_nodes)
    print(model)
    dummy_input = torch.randn(32, 3, num_nodes, 12)
    dummy_adj = torch.eye(num_nodes)
    L = get_laplacian(dummy_adj)
    output = model(dummy_input, L)
    print(f"输入形状：{dummy_input.shape}")
    print(f"输出形状：{output.shape}")