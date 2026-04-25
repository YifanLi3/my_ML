import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def dm01():
    img = plt.imread("./img/a.jpg")

    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute(2, 0, 1)