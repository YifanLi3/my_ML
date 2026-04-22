from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt

def prepare_data():
    X = np.array([50,60,70,80,90,100,110,120])
    y = np.array([26.5,31.8,36.2,41.5,46.8,52.1,57.8,62.7])

    return X, y

def plot_linear_regression(X, y):
    X_line = np.linspace(45, 125, 100)
    y_ideal = 0.5 * X_line + 1.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='red', s=60, label='True(Noise)')
    ax.plot(X_line, y_ideal, color='blue', linestyle='-', linewidth=2, label='y = 0.5x + 1.2')

    ax.set_xlabel('Area', fontsize=20)
    ax.set_ylabel('Price', fontsize=20)
    ax.set_title('Linear Regression', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    plt.show()


if __name__ == '__main__':
    X, y = prepare_data()
    plot_linear_regression(X, y)