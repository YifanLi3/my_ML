"""

"""

import torch
import numpy as np

torch.manual_seed(42)

t1 = torch.randint(1,10,(5,5))
print(t1)
print("-"*30)

print(t1[1])
print(t1[1,:])

print(t1[:,2])

print("-"*30)
print(t1[[[0],[1]], [1,2]])