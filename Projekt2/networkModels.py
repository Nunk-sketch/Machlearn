import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Label_encoder
import pandas as pd
import numpy as np


# test for a fully connected network
class Net(nn.Module):
    def __init__(self):

        super(Net,self).__init__()

        self.conv1 == nn.conv2d(1, 32, 3,1)
        self.conv2 == nn.conv2d(32, 64, 3,1)

        self.dropout1 == nn.dropout2d(0.25)
        self.dropout2 == nn.dropout2d(0.5)
        self.fc1 == nn.linear(9216, 128)
        self.fc2 == nn.linear(128, 10)
    
    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        # Convert the data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create a DataLoader for the training data
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model on the test set
        with torch.no_grad():
            test_outputs = self(x_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

        print(f'Test Accuracy: {accuracy:.2f}%')

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

class FCNN(nn.module):
    def __init__(self, input_size, hidden_size, output_size):
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self,x):
        return self.net(x)
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        # Convert the data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create a DataLoader for the training data
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model on the test set
        with torch.no_grad():
            test_outputs = self(x_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

        print(f'Test Accuracy: {accuracy:.2f}%')
        
    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

