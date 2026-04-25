import numpy as np
import matplotlib.pyplot as plt
import torch

def dm01():
    img1 = np.zeros((200, 200, 3))
    print(f"img1{img1}")
    #plt.imshow(img1)
    #plt.show()

    img2 = torch.full((200, 200, 3), 255)
    plt.imshow(img2)
    plt.show()

if __name__ == "__main__":
    dm01()