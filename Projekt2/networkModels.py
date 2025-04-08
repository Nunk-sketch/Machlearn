import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# test for a fully connected network
# class Net(nn.Module):
#     def __init__(self):

#         super(Net,self).__init__()

#         self.conv1 == nn.conv2d(1, 32, 3,1)
#         self.conv2 == nn.conv2d(32, 64, 3,1)

#         self.dropout1 == nn.dropout2d(0.25)
#         self.dropout2 == nn.dropout2d(0.5)
#         self.fc1 == nn.linear(9216, 128)
#         self.fc2 == nn.linear(128, 10)
    
#     def forward(self,x):

#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)

#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)

#         x = self.fc2(x)

#         return F.log_softmax(x, dim=1)
    
#     def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
#         # Convert the data to PyTorch tensors
#         x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#         x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
#         y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#         # Create a DataLoader for the training data
#         train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#         # Define the loss function and optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(self.parameters(), lr=0.001)

#         # Training loop
#         for epoch in range(epochs):
#             for inputs, labels in train_loader:
#                 optimizer.zero_grad()
#                 outputs = self(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

#         # Evaluate the model on the test set
#         with torch.no_grad():
#             test_outputs = self(x_test_tensor)
#             _, predicted = torch.max(test_outputs.data, 1)
#             accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

#         print(f'Test Accuracy: {accuracy:.2f}%')

#     def predict(self, x):
#         x_tensor = torch.tensor(x, dtype=torch.float32)
#         with torch.no_grad():
#             outputs = self(x_tensor)
#             _, predicted = torch.max(outputs.data, 1)
#         return predicted.numpy()

class FCNN(nn.Module):
    def __init__(self):

        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  
            nn.ReLU(),           # Activation function
            nn.Linear(64, 128),  
            nn.ReLU(),           # Activation function
            nn.Linear(128, 64),   
            nn.ReLU(),           # Activation function
            nn.Linear(64, 8),
            nn.ReLU(),           # Activation function
            nn.Linear(8, 3),    # Output layer)
            
        )

    def forward(self,x):
        return self.net(x)
    
    def compute_loss(self, x, y):
        """
        Evaluate the loss of the model on the given data.

        Parameters:
        x (array-like): Input features.
        y (array-like): Target labels.

        Returns:
        float: Computed loss.
        """
        # Convert the data to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Ensure the model is in evaluation mode
        self.eval()

        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Compute the loss
        with torch.no_grad():
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)

        return loss.item()

    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        # Check and convert the data to numpy arrays if they are pandas DataFrames
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.to_numpy()
        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.to_numpy()
        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        # Convert the data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Ensure target labels are within the valid range
        num_classes = 3  # Update this if the number of classes changes
        if y_train_tensor.max() >= num_classes or y_test_tensor.max() >= num_classes:
            raise ValueError(f"Target labels must be in the range [0, {num_classes - 1}]. Found out-of-bounds labels.")

        # Create a DataLoader for the training data
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_history = []
        val_loss_history = []

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            self.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

            # Save training loss
            avg_train_loss = total_loss / len(train_loader)
            loss_history.append(avg_train_loss)

            # Evaluate the model on the test set
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                test_outputs = self(x_test_tensor)
                val_loss = criterion(test_outputs, y_test_tensor).item()
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%')

            # Save validation loss
            val_loss_history.append(val_loss)

        # Save the model and loss history
        print("Saving model and loss.")
        torch.save(self.state_dict(), "FCNN_model_raw.pth")
        np.save("loss_history.npy", np.array(loss_history, dtype=np.float32))
        np.save("val_loss_history.npy", np.array(val_loss_history, dtype=np.float32))
        print("Done saving model and loss.")
    

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
    
    def fit(self, X, y, epochs=10, batch_size=32):
        """
        Fit the model to the provided data.

        Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.

        Returns:
        self: Trained model.
        """
        # Check and convert the data to numpy arrays if they are pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        # Convert the data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Ensure target labels are within the valid range
        num_classes = 3  # Update this if the number of classes changes
        if y_tensor.max() >= num_classes:
            raise ValueError(f"Target labels must be in the range [0, {num_classes - 1}]. Found out-of-bounds labels.")

        # Create a DataLoader for the training data
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            self.train()  # Set the model to training mode
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(data_loader):.4f}')

        return self
    

