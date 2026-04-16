
import torch
import numpy as np

def demo02():
    t1 = torch.randint(1, 10, (2,3))
    print(f't1:{t1}, type:{type(t1)}, {t1.shape}')
    t2 = t1.unsqueeze(1)
    print(f't1:{t2}, type:{type(t2)}, {t2.shape}')

def demo03():
    t1 = torch.randint(1, 10, (2,3,4))
    print(f't1:{t1}, type:{type(t1)}, {t1.shape}')

    t2 = t1.transpose(0, -1)
    print(f't2:{t2}, type:{type(t2)}, {t2.shape}')

    t3 = t1.permute(2, 0, 1)
    print(f't3:{t3}, type:{type(t3)}, {t3.shape}')




if __name__ == "__main__":
    #demo01()
    demo03()

