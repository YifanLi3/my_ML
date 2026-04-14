import torch
import numpy as np

def demo01():
    t1 = torch.tensor([1,2,3,4,5,6])
    print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')
    print("-"*30)

    #n1 = t1.numpy()
    n1 = t1.numpy().copy()
    print(f'n1:{n1}, type:{type(n1)}, {n1.dtype}')

    n1[0] = 100
    print(f'n1:{n1}, type:{type(n1)}, {n1.dtype}')
    print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')
    print("-"*30)

def demo02():
    n1 = np.array([1,2,3,4,5,6])
    print(f'n1:{n1}, type:{type(n1)}, {n1.dtype}')

    #t1 = torch.from_numpy(n1)
    t1 = torch.tensor(n1)
    print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')

    n1[0] = 100
    print(f'n1:{n1}, type:{type(n1)}, {n1.dtype}')
    print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')


def demo03():
    t1 = torch.tensor(100)
    #t1 = torch.tensor([100, ])
    print(f't1:{t1}, type:{type(t1)}, {t1.dtype}')

    a = t1.item()
    print(f'a:{a}, type:{type(a)}')

    


if __name__ == "__main__":
    #demo01()
    #demo02()
    demo03()

