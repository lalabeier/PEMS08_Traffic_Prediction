import sys
sys.path.append('./code')
from model import STGCN, get_laplacian
import torch

num_nodes = 170
model = STGCN(num_nodes)
dummy_input = torch.randn(32, 3, num_nodes, 12)
dummy_adj = torch.eye(num_nodes)
L = get_laplacian(dummy_adj)
output = model(dummy_input, L)
print("Test passed, output shape:", output.shape)