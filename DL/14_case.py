import torch

x = torch.ones(2,5)
print(f"x:{x}")

y = torch.zeros(2,3)
print(f"y:{y}")

w = torch.randn(5,3, requires_grad=True)
print(f"w:{w}")

b = torch.randn(3, requires_grad=True)
print(f"b:{b}")

z = torch.matmul(x,w) + b

criterion = torch.nn.MSELoss()
loss = criterion(z, y)
print(f"loss:{loss}")

loss.backward()
print(f"w.grad:{w.grad}")
print(f"b.grad:{b.grad}")

