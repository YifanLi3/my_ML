from sched import scheduler
import torch
from torch import optim
import matplotlib.pyplot as plt

def dm01():

    lr, epochs, iterations = 0.1, 200, 10

    y_true = torch.tensor([0])
    x = torch.tensor([1.0], dtype=torch.float32)

    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    lr_list, epoch_list = [], []

    for epoch in range(epochs):

        epoch_list.append(epoch)

        lr_list.append(schedular.get_last_lr())

        for batch in range(iterations):
            y_pred = w * x 
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    print(f"lr_list: {lr_list}")

    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

if __name__ == '__main__':
    dm01()