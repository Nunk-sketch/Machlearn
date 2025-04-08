import numpy as np
import matplotlib.pyplot as plt
from networkModels import FCNN
from abaclass import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def visualize_loss(train_loss_history: np.ndarray, val_loss_history: np.ndarray) -> None:
    """Visualize the training and validation loss."""
    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="Training loss")
    plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


train_loss_history = np.load("loss_history.npy", allow_pickle=True)
val_loss_history = np.load("val_loss_history.npy", allow_pickle=True)

visualize_loss(train_loss_history, val_loss_history)

