import torch
x = torch.rand((10000, 10000), device='cuda')
y = torch.mm(x, x)
print("Matrix multiplication complete.")