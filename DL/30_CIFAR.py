from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

BATCH_SIZE = 8

def create_dataset():
    train = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    valid = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    return train, valid


if __name__ == '__main__':
    train_dataset, valid_dataset = create_dataset()
    print(train_dataset.class_to_idx)