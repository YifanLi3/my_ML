import torch, numpy

t1 = torch.tensor([1,3,4], dtype=torch.float)
print(f't1{t1}, type:{type(t1)}, {t1.dtype}')
print("-"*30)

t2 = t1.type(torch.int16)
print(f't1{t2}, type:{type(t2)}, {t2.dtype}')