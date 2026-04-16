import torch

t1 = torch.tensor([10,20], requires_grad=True, dtype=torch.float)
print(t1)

n1 = t1.detach()
print(n1)

t1.data[0] = 100
print(t1)
print(n1)

print(f"t1:{t1.requires_grad}")

print(f"n1:{n1.requires_grad}")

n2 = n1.numpy()
print(n2)
