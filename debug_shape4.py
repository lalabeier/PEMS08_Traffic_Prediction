import sys
sys.path.append('./code')
from preprocess import load_raw_data, split_train_test, normalize, create_sequences
import numpy as np
import torch

print("=== Full preprocessing ===")
data = load_raw_data('./data/PEMS08_raw/')
train_data, test_data = split_train_test(data)
print(f"Train data shape: {train_data.shape}")
train_norm, mean, std = normalize(train_data)
test_norm, _, _ = normalize(test_data, mean, std)
X_train, y_train = create_sequences(train_norm, seq_len=12, pred_len=1)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_train dims: samples={X_train.shape[0]}, nodes={X_train.shape[1]}, seq_len={X_train.shape[2]}, features={X_train.shape[3]}")
print(f"y_train dims: samples={y_train.shape[0]}, nodes={y_train.shape[1]}, pred_len={y_train.shape[2]}, features={y_train.shape[3]}")

# Permute as in main.py
X = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
y = torch.FloatTensor(y_train).permute(0, 3, 1, 2)
print(f"After permute X shape: {X.shape}")
print(f"After permute y shape: {y.shape}")
print(f"Interpretation: X batch={X.shape[0]}, channels={X.shape[1]}, nodes={X.shape[2]}, timesteps={X.shape[3]}")
print(f"Interpretation: y batch={y.shape[0]}, channels={y.shape[1]}, nodes={y.shape[2]}, pred_len={y.shape[3]}")

# Check num_nodes
num_nodes = X.shape[2]
print(f"num_nodes extracted: {num_nodes}")

# Quick model forward pass
sys.path.append('./code')
from model import STGCN, get_laplacian
model = STGCN(num_nodes, in_channels=3, hidden_channels=64, out_channels=1, K=3)
adj = torch.eye(num_nodes)
L = get_laplacian(adj)
output = model(X[:2], L)  # small batch
print(f"Model output shape: {output.shape}")
print(f"Expected y shape: {y[:2].shape}")
print("If shapes match, training should work.")