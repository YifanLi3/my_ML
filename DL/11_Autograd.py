import torch
import numpy as np

w = torch.tensor(10, requires_grad=True, dtype=torch.float)
print(f"w: {w}, requires_grad: {w.requires_grad}")

loss = 2 * w ** 2

#loss.sum().backward()
loss.backward()

w.data = w.data - 0.01 * w.grad
print(f"w: {w}, requires_grad: {w.requires_grad}")