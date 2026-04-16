import torch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)

x = torch.linspace(-20, 20, 1000)
print(f"x: {x}")

y = torch.sigmoid(x)

axes[0].plot(x,y)
axes[0].set_title('Sigmoid Plot')
axes[0].grid()

x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.sigmoid(x).sum().backward()

axes[1].plot(x.detach(), x.grad)
axes[1].set_title('Sigmoid Derivative Plot')
axes[1].grid()

plt.show()