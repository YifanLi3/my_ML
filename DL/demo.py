import torch
import matplotlib.pyplot as plt

TEMPERATURE = 30

def ema():
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

    print (isinstance(ema, list))
    print (isinstance(temper, list))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(date, temper, label="Temperature")
    ax1.scatter(date, temper)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Temperature")
    ax1.set_title("Random Temperature")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(date, ema, label="EMA Temperature")
    ax2.scatter(date, ema)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Temperature")
    ax2.set_title("EMA Temperature")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.show()

def momentum():
    


if __name__ == '__main__':
    ema()
