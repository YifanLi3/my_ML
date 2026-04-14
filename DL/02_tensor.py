import torch, numpy

t1 = torch.ones(2,3)
print(f't1{t1}, type:{type(t1)}')
print("-"*30)

t2 = torch.tensor([[1,2], [3,4], [5,6]])
print(f't1{t2}, type:{type(t2)}')
print("-"*30)

t3 = torch.ones_like(t2)
print(f't1{t3}, type:{type(t3)}')
print("-"*30)

t4 = torch.zeros(2,3)
print(f't1{t4}, type:{type(t4)}')
print("-"*30)

t5 = torch.tensor([[1,2], [3,4], [5,6]])
print(f't1{t5}, type:{type(t5)}')
print("-"*30)

t6 = torch.zeros_like(t5)
print(f't1{t6}, type:{type(t6)}')
print("-"*30)