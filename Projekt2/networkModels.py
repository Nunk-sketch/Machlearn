import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


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
            nn.ReLU(),           
            nn.Linear(64, 128),  
            nn.ReLU(),           
            nn.Linear(128, 64),   
            nn.ReLU(),           
            nn.Linear(64, 8),
            nn.ReLU(),           
            nn.Linear(8, 3),    
        )

    def forward(self, x):
        return self.net(x)
    
    def compute_loss(self, x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        self.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = self(x_tensor)
            loss = criterion(outputs, y_tensor)
        return loss.item()

    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32, use_cross_validation=False, cv_folds=5):
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.to_numpy()
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.to_numpy()
        if isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_test = y_test.to_numpy()

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        num_classes = 3
        if y_train_tensor.max() >= num_classes or y_test_tensor.max() >= num_classes:
            raise ValueError(f"Target labels must be in the range [0, {num_classes - 1}]. Found out-of-bounds labels.")

        if use_cross_validation:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            best_model_state = None
            best_accuracy = 0

            for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
                print(f"Starting fold {fold + 1}/{cv_folds}")
                x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                x_train_fold_tensor = torch.tensor(x_train_fold, dtype=torch.float32)
                y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.long)
                x_val_fold_tensor = torch.tensor(x_val_fold, dtype=torch.float32)
                y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.long)

                train_dataset = TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self.parameters(), lr=0.001)

                for epoch in range(epochs):
                    self.train()
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                self.eval()
                with torch.no_grad():
                    val_outputs = self(x_val_fold_tensor)
                    _, predicted = torch.max(val_outputs.data, 1)
                    accuracy = (predicted == y_val_fold_tensor).sum().item() / len(y_val_fold_tensor) * 100

                print(f"Fold {fold + 1} Validation Accuracy: {accuracy:.2f}%")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = self.state_dict()

            print(f"Best Cross-Validation Accuracy: {best_accuracy:.2f}%")
            self.load_state_dict(best_model_state)

        else:
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.parameters(), lr=0.001)

            for epoch in range(epochs):
                self.train()
                total_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

            self.eval()
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
