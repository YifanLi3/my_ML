import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import linear, dropout

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
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    num_epochs = 4
    for epoch in range(num_epochs):
        criterion = (w**2)/2.0
        optimizer.zero_grad()
        criterion.sum().backward()
        optimizer.step()

        print(f"w:{w}, w.grad:{w.grad}")

def adagrad():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.Adagrad(params=[w], lr = 0.01)

    num_epochs = 4
    for epoch in range(num_epochs):
        criterion = (w**2)/2.0
        optimizer.zero_grad()
        criterion.sum().backward()
        optimizer.step()

        print(f"w:{w}, w.grad:{w.grad}")

def rmsprop():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    num_epochs = 4
    for epoch in range(num_epochs):
        criterion = (w**2)/2.0
        optimizer.zero_grad()
        criterion.sum().backward()
        optimizer.step()

        print(f"w:{w}, w.grad:{w.grad}")

def adam():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))

    num_epochs = 4
    for epoch in range(num_epochs):
        criterion = (w**2)/2.0
        optimizer.zero_grad()
        criterion.sum().backward()
        optimizer.step()

        print(f"w:{w}, w.grad:{w.grad}")

def dropout():
    x = torch.randint(1, 10, [1, 4]).float()
    print(x)
    linear1 = nn.Linear(4, 5)
    l = linear1(x)
    print(l)
    y = torch.relu(l)
    print(y)
    dropout = nn.Dropout(p=0.5)
    output = dropout(y)
    print (output)


if __name__ == '__main__':
    #ema()
    #momentum()
    #adagrad()
    #rmsprop()
    #adam()
    dropout()


