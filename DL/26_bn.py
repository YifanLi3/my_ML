import torch
import torch.nn as nn

def dm01():
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print(input_2d)

    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1, affine=True)

    output = bn2d(input_2d)
    print(output)

def dm02():
    input_1d = torch.randn(size=(2, 2))
    print(input_1d)

    linear1 = nn.Linear(2, 4)
    l1 = linear1(input_1d)

    bn1d = nn.BatchNorm1d(num_features=4)
    output_1d = bn1d(l1)
    print(output_1d)

if __name__ == '__main__':
    #dm01()
    dm02()