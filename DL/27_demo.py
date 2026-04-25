from turtle import forward
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchsummary import summary

def create_dataset():
    data = pd.read_csv("./data/phone.csv")
    print(data.head())
    print(data.shape)

    x,y = data.iloc[:, :-1], data.iloc[:, -1]


    train_dataset = TensorDataset(
                    torch.tensor(x_train.values),
                    torch.tensor(y_train.values)
                    )

    test_dataset = TensorDataset(
                    torch.tensor(x_test.values),
                    torch.tensor(y_test.values)
                    )
    return train_dataset, test_dataset, x_test.shape[1], len(np.unique(y))

class PhonePriceModel(nn.module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = self.output(x)

        return x

def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 50
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x,y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        print(f"epoch {epoch+1}, loss: {total_loss/batch_num}")


if __name__ == "__main__":
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    model = PhonePriceModel(input_dim, output_dim)
    summary(model, input_size=(16, input_dim))