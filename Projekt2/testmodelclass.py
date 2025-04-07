import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from abaclass import *
from networkModels import FCNN


model = FCNN()
model.load_state_dict(torch.load("FCNN_model_raw.pth", map_location=torch.device("cpu")))

model.eval()

# Convert the test data to PyTorch tensors
x_test_tensor = torch.tensor(x_test_mat_clas, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_mat_clas, dtype=torch.long)

# Create a DataLoader for the test data
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test data
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate average loss and accuracy
test_loss /= len(test_loader)
accuracy = correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

