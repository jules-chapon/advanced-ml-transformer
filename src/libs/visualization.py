"""Visualization functions"""

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses: np.ndarray, valid_losses: np.ndarray) -> None:
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.title("Training and Validation Losses")
    plt.grid(visible=True, which="major", axis="y")
    plt.xlabel("Epoch")
    plt.xticks([x for x in range(0, train_losses.shape[0] - 1) if x % 10 == 0])
    plt.ylabel("Loss")
    plt.ylim((0, 10))
    plt.legend(loc="upper right")
    plt.show()
