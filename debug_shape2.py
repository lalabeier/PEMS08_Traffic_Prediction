import sys
sys.path.append('./code')
from preprocess import load_raw_data, split_train_test, normalize, create_sequences
import numpy as np

print("Testing data shapes...")
data = load_raw_data('./data/PEMS08_raw/')
print(f"Raw data shape: {data.shape}")
print(f"Data dimensions interpretation: nodes={data.shape[0]}, timesteps={data.shape[1]}, features={data.shape[2]}")
train_data, test_data = split_train_test(data)
print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
print(f"Train data dimensions: nodes={train_data.shape[0]}, timesteps={train_data.shape[1]}, features={train_data.shape[2]}")
train_norm, mean, std = normalize(train_data)
X_train, y_train = create_sequences(train_norm, seq_len=12, pred_len=1)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_train dimensions: samples={X_train.shape[0]}, nodes={X_train.shape[1]}, seq_len={X_train.shape[2]}, features={X_train.shape[3]}")
print(f"y_train dimensions: samples={y_train.shape[0]}, nodes={y_train.shape[1]}, pred_len={y_train.shape[2]}, features={y_train.shape[3]}")

# Now permute as in main.py (0,3,1,2)
import torch
X = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
print(f"After permute shape: {X.shape}")
print(f"Interpretation: batch={X.shape[0]}, channels={X.shape[1]}, nodes={X.shape[2]}, timesteps={X.shape[3]}")
# Check if nodes equals 170 or something else
print(f"Nodes dimension size: {X.shape[2]}")
print(f"Timesteps dimension size: {X.shape[3]}")

# Also permute y
y = torch.FloatTensor(y_train).permute(0, 3, 1, 2)
print(f"y after permute shape: {y.shape}")
print(f"Interpretation: batch={y.shape[0]}, channels={y.shape[1]}, nodes={y.shape[2]}, pred_len={y.shape[3]}")