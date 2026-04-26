import torch
import matplotlib.pyplot as plt

TEMPERATURE = 30

def demo01():
    # stable random number
    torch.manual_seed(42)

    # y axis
    temper = torch.randn(TEMPERATURE) * 10
    # x axis
    date = torch.arange(1, TEMPERATURE+1, 1)

    sum = 0
    beta = 0.9

    ema = []
    for i, temp in enumerate(temper):
        if i == 0:
            ema.append(temp)
            continue
        sum = ema[i-1] * beta + (1 - beta) * temp
        ema.append(sum)


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(date, temper, label="Temperature")
    ax.scatter(date, temper)
    ax.set_xlabel("Day")
    ax.set_ylabel("Temperature")
    ax.set_title("Random Temperature")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.show()

if __name__ == '__main__':
    demo01()
