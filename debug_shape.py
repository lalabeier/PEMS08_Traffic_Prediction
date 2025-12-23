import sys
sys.path.append('.')
from preprocess import load_raw_data, split_train_test, normalize, create_sequences
import numpy as np

print("Testing data shapes...")
data = load_raw_data('./data/PEMS08_raw/')
print(f"Raw data shape: {data.shape}")
train_data, test_data = split_train_test(data)
print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
train_norm, mean, std = normalize(train_data)
test_norm, _, _ = normalize(test_data, mean, std)
print(f"Train norm shape: {train_norm.shape}")
X_train, y_train = create_sequences(train_norm, seq_len=12, pred_len=1)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_train dimensions: samples={X_train.shape[0]}, nodes={X_train.shape[1]}, seq_len={X_train.shape[2]}, features={X_train.shape[3]}")
print(f"y_train dimensions: samples={y_train.shape[0]}, nodes={y_train.shape[1]}, pred_len={y_train.shape[2]}, features={y_train.shape[3]}")

# Now permute as in main.py
import torch
X = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
print(f"After permute shape: {X.shape}")
print(f"Interpretation: batch={X.shape[0]}, channels={X.shape[1]}, nodes={X.shape[2]}, timesteps={X.shape[3]}")
