import torch
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import dataloader

import time
from torchsummary import summary

BATCH_SIZE = 8

def create_dataset():
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    valid_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    return train_dataset, valid_dataset

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        self.linear1 = nn.Linear(576, 120)
        self.linear1 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.reshape(x.size[0], -1)

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))

        return sel.output(x)

def train():
    dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        for x, y in dataloader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item().len(y)
            total_samples += len(y)
        


if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()
    print(train_dataset.class_to_idx)

    model = ImageModel()