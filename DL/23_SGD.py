import torch
import torch.optim as optim

def dm01_momentum():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    criterion = ((w**2)/2.0)

    optimizer = optim.SGD(params=[w], lr = 0.01, momentum=0.9)

    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

def dm02_adagrad():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    criterion = ((w**2)/2.0)

    optimizer = optim.Adagrad(params=[w], lr = 0.01)

    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

def dm03_rmsprop():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    criterion = ((w**2)/2.0)

    optimizer = optim.RMSprop(params=[w], lr = 0.01, alpha=0.99)

    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

def dm04_adam():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    criterion = ((w**2)/2.0)

    optimizer = optim.Adam(params=[w], lr = 0.01, betas=(0.9, 0.999))

    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

    criterion = ((w**2)/2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f"w:{w}, w.grad:{w.grad}")

if __name__ == '__main__':
    #dm01_momentum()
    #dm02_adagrad()
    #dm03_rmsprop()
    dm04_adam()

