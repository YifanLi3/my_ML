import torch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2)

x = torch.linspace(-20, 20, 1000)
y = torch.relu(x)

axes[0].plot(x,y)
axes[0].set_title('ReLu Plot')
axes[0].grid('x')


x = torch.linspace(-20, 20, 1000, requires_grad=True)

torch.relu(x).sum().backward()

axes[1].plot(x.detach(), x.grad)
axes[1].set_title('ReLu Derivative Plot')
axes[1].grid('x')

plt.show()