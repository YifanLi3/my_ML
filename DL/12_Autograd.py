import torch

w = torch.tensor(10, requires_grad=True, dtype=torch.float)
print(f"w: {w}, requires_grad: {w.requires_grad}")

loss = w ** 2 + 20


for i in range(1,101):
    loss = w ** 2 + 20

    if w.grad is not None:
        w.grad.zero_()

    loss.sum().backward()

    w.data = w.data - 0.02 * w.grad

    print(f"{i}th: w: {w}, loss: {loss}")

print(f"Final: w: {w}, grad: {w.grad}, loss: {loss}")

