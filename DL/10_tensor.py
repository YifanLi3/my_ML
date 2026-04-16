import torch
import numpy as np

torch.manual_seed(13)

t1 = torch.randint(1, 10, (2,3))
print(f't1:{t1}, type:{type(t1)}, {t1.shape}')

t2 = torch.randint(1, 10, (2,3))
print(f't1:{t2}, type:{type(t2)}, {t2.shape}')

t3 = torch.cat([t1, t2], dim=0)
print(f't3:{t3}, type:{type(t3)}, {t3.shape}')

t4 = torch.cat([t1, t2], dim=1)
print(f't4:{t4}, type:{type(t4)}, {t4.shape}')

t5 = torch.stack([t1, t2], dim=0)
print(f't4:{t5}, type:{type(t5)}, {t5.shape}')

t6 = torch.stack([t1, t2], dim=2)
print(f't4:{t6}, type:{type(t6)}, {t6.shape}')