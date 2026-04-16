import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# y = coef * x + 14.5 + 随机噪声
def create_dataset():
    x,y,coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        coef=True,
        bias=14.5,
        random_state=42
    )
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x,y,coef

def train(x,y,coef):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = nn.Linear(1,1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0

    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y.reshape(-1,1))
            total_loss += loss.item()
            total_sample += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(total_loss/total_sample)

        print(f"Epoch:{epoch+1}, Average Loss: {total_loss/total_sample}")

    print(f"Epoch:{epoch}, Average Loss: {loss_list}")

    plt.plot(range(epochs), loss_list)
    plt.title("Loss Curve")
    plt.grid()
    plt.show()

    plt.scatter(x,y)
    w = model.weight.detach().item()
    b = model.bias.detach().item()

    y_pred = torch.tensor([v*w+b for v in x])
    y_true = torch.tensor([v*coef+14.5 for v in x])
    plt.plot(x, y_pred, color='red', label='Prediction')
    plt.plot(x, y_true, color='blue', label='True')

    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x,y,coef = create_dataset()
    train(x,y,coef)





