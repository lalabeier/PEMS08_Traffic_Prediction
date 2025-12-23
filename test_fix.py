import sys
sys.path.append('./code')
from preprocess import load_raw_data, split_train_test, normalize, create_sequences
from model import STGCN, get_laplacian
import torch
import numpy as np

print("Testing fixed model...")
data = load_raw_data('./data/PEMS08_raw/')
train_data, test_data = split_train_test(data)
train_norm, mean, std = normalize(train_data)
X_train, y_train = create_sequences(train_norm, seq_len=12, pred_len=1)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Permute as in main.py (0,3,1,2)
X = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
y = torch.FloatTensor(y_train).permute(0, 3, 1, 2)
print(f"X shape after permute: {X.shape}")
print(f"y shape after permute: {y.shape}")

num_nodes = X.shape[2]
print(f"num_nodes: {num_nodes}")

# Create model with out_channels=3
model = STGCN(num_nodes, in_channels=3, hidden_channels=64, out_channels=3, K=3)
adj = torch.eye(num_nodes)
L = get_laplacian(adj)
output = model(X[:2], L)
print(f"Model output shape: {output.shape}")
print(f"Target shape: {y[:2].shape}")

# Check if shapes match
if output.shape == y[:2].shape:
    print("SUCCESS: Shapes match!")
else:
    print("FAIL: Shapes mismatch.")
    print(f"Output: {output.shape}, Target: {y[:2].shape}")

# Test loss computation
criterion = torch.nn.MSELoss()
loss = criterion(output, y[:2])
print(f"Loss computed: {loss.item()}")

# Test with batch size 8 (as in warning)
X_batch = X[:8]
y_batch = y[:8]
output_batch = model(X_batch, L)
print(f"Batch output shape: {output_batch.shape}")
print(f"Batch target shape: {y_batch.shape}")
if output_batch.shape == y_batch.shape:
    print("Batch shapes match.")
else:
    print("Batch shapes mismatch.")