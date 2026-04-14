import torch
import numpy as np

t1 = torch.tensor([1,2,3,4,5,6])
t2 = t1.add(10)
t2 = t1 + 10
print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')
print(f't2:{t2}, type:{type(t2)}, {t2.dtype}')