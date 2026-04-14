"""
torch.tensor
torch.Tensor
torch.IntTensor
torch.Size
torch.FloatTensor
"""
import torch
import numpy as np

# 1. define function
def demo01():
    # 1.1
    t1 = torch.tensor(10)
    print(f't1{t1}, type:{type(t1)}')

    # 1.3
    data = np.random.randint(0, 10, (2,3))
    t3 = torch.tensor(data, dtype=torch.float)
    print(f't1{t3}, type:{type(t3)}')

    # 1.4

def demo02():
    t1 = torch.Tensor(10)
    print(f't1{t1}, type:{type(t1)}')
    print("-"*30)

    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f't1{t2}, type:{type(t2)}')
    print("-"*30)

    data = np.random.randint(0, 10, (2,3))
    t3 = torch.Tensor(data)
    print(f't1{t3}, type:{type(t3)}')
    print("-"*30)

    t4 = torch.Tensor(2,3)
    print(f't1{t4}, type:{type(t4)}')

def demo03():
    t1 = torch.IntTensor(10)
    print(f't1{t1}, type:{type(t1)}')
    print("-"*30)

    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f't1{t2}, type:{type(t2)}')
    print("-"*30)

    data = np.random.randint(0, 10, (2,3))
    t3 = torch.IntTensor(data)
    print(f't1{t3}, type:{type(t3)}')
    print("-"*30)

    data = np.random.randint(0, 10, (2,3))
    t3 = torch.IntTensor(data)
    print(f't1{t3}, type:{type(t3)}')
    print("-"*30)

    data = np.random.randint(0, 10, (2,3))
    t3 = torch.FloatTensor(data)
    print(f't1{t3}, type:{type(t3)}')
    print("-"*30)



if __name__ == "__main__":
    #demo01()
    #demo02()
    demo03()