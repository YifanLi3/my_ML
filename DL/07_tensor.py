"""
diancheng
t1 * t2
t1.mul(t2)

juzhencheng
t1 @ t2
t1.matmul(t2)
"""

import torch
import numpy as np

def demo01():
    t1 = torch.tensor([[1,2,3],[4,5,6]])

    t2 = torch.tensor([[5,6,7],[8,9,10]])

    t3 = t1 * t2
    #t3 = t1.mul(t2)

    print(t3)

def demo02():
    t1 = torch.tensor([
            [1,2,3],
            [4,5,6]
            ])

    t2 = torch.tensor([
                [5,6],
                [8,9],
                [10,11]
                ])

    #t3 = t1.matmul(t2)
    t3 = t1 @ t2

    print(t3)


if __name__ == "__main__":
    #demo01()
    demo02()
    #demo03()