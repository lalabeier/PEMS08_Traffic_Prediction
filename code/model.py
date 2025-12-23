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

class SpatialAttention(nn.Module):
    """空间注意力模块（多头自注意力）"""
    def __init__(self, channels, nodes, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.nodes = nodes
        self.channels = channels

    def forward(self, x):
        """
        x: (batch, channels, nodes, timesteps)
        输出：经过注意力加权的特征，形状相同
        """
        batch, channels, nodes, timesteps = x.shape
        # 在时间维度上平均池化，得到节点特征 (batch, channels, nodes)
        x_pool = x.mean(dim=-1)  # (batch, channels, nodes)
        x_pool = x_pool.permute(0, 2, 1)  # (batch, nodes, channels)
        # 计算Q,K,V
        qkv = self.to_qkv(x_pool).reshape(batch, nodes, 3, self.heads, channels // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, nodes, channels_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(batch, nodes, channels)
        out = self.to_out(out)  # (batch, nodes, channels)
        out = out.permute(0, 2, 1)  # (batch, channels, nodes)
        out = out.unsqueeze(-1).expand_as(x)  # (batch, channels, nodes, timesteps)
        return x + out  # 残差连接

class STGCN(nn.Module):
    """完整的STGCN模型"""
    def __init__(self, num_nodes, in_channels=3, hidden_channels=64, out_channels=1, K=3, num_blocks=2,
                 use_attention=False, attention_heads=4, attention_dropout=0.1):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.use_attention = use_attention
        self.blocks = nn.ModuleList()
        self.blocks.append(STGCNBlock(in_channels, hidden_channels, K))
        for _ in range(num_blocks - 1):
            self.blocks.append(STGCNBlock(hidden_channels, hidden_channels, K))
        # 注意力模块
        self.attentions = nn.ModuleList()
        if use_attention:
            for _ in range(num_blocks):
                self.attentions.append(SpatialAttention(hidden_channels, num_nodes, attention_heads, attention_dropout))
        else:
            self.attentions = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.final_temp = TemporalConv(hidden_channels, hidden_channels)
        self.output = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x, L):
        """
        x: (batch, channels, nodes, timesteps)
        L: 拉普拉斯矩阵 (nodes, nodes)
        """
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x, L)
            x = attention(x)
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