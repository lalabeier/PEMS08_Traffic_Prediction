import sys
sys.path.append('./code')
from preprocess import load_raw_data
import numpy as np

data = load_raw_data('./data/PEMS08_raw/')
print("Final data shape:", data.shape)
print("Nodes:", data.shape[0], "Timesteps:", data.shape[1], "Features:", data.shape[2])