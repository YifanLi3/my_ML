import torch
import torch.nn as nn

def dm01():
    t1 = torch.randint(0, 10, size=(1,4)).float()
    print(t1)

    linear1 = nn.Linear(4, 5)

    l1 = linear1(t1)
    output = torch.relu(l1)
    print(output)
    dropout = nn.Dropout(p=0.5)
    d1 = dropout(output)
    print(d1)


if __name__ == '__main__':
    dm01()