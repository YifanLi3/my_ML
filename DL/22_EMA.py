import torch

import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30

def demo01():
    torch.manual_seed(42)

    tem = torch.randn(size = [ELEMENT_NUMBER,]) * 10

    print (tem)

    days = torch.arange(1, ELEMENT_NUMBER+1, 1)

    plt.plot(days, tem, color='r')
    plt.scatter(days, tem)
    plt.show()

def demo02(beta=0.9):
    torch.manual_seed(42)

    tem = torch.randn(size = [ELEMENT_NUMBER,]) * 10

    print (tem)

    days = torch.arange(1, ELEMENT_NUMBER+1, 1)

    exp_weight_avg = []
    for idx, temp in enumerate(tem, 1):
        if idx == 1:
            exp_weight_avg.append(temp)
            continue

        new_temp = exp_weight_avg[idx-2] * beta + (1 - beta) * temp
        exp_weight_avg.append(new_temp)

    plt.plot(days, exp_weight_avg, color='r', label=f'EMA (beta={beta})')
    plt.scatter(days, tem, label='Original')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #demo01()
    demo02(0.9)