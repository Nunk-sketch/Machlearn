import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from abaclass import *
from networkModels import FCNN


class EarlyStopping:
    def __init__(self, patience=10) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.rewind = False
        self.rewinded = False

    def check(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(
    model: FCNN,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: torch.optim.Adam,
    criterion: nn.CrossEntropyLoss,
    epochs: int,
    device: torch.device
    ) -> None:
    """Trains the FCNN model."""
    # Initialize early stopping.
    early_stopping = EarlyStopping()

    # Lists for tracking loss
    loss_history = []
    val_loss_history = []

    # Training loop.
    for epoch in range(epochs):
    # Set the model to training mode.
        model.train()

    # Keeps track of the loss.
        total_loss = 0.0

    # Fetches a sample from the dataloader which is an instance of 'DataLoader'.
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Move data to the specified device.
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients from previous iteration.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(inputs)

            # Compute the loss.
            loss = criterion(outputs, targets)
        
            # Backward pass and optimization.
            loss.backward()
            optimizer.step()

            # Update the total loss.
            total_loss += loss.item()

            print(f"Batch {i + 1}/{len(train_dataloader)} processed.")
    
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")
        loss_history.append(avg_train_loss)

    # Set model to evaluation mode.
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(validation_dataloader):
            # Move data to the specified device.
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass.
            outputs = model(inputs)

            # Compute the loss.
            loss = criterion(outputs, targets)

            val_loss += loss.item()
    
            print(f"Validation Batch {i + 1}/{len(validation_dataloader)} processed.")

        avg_val_loss = val_loss / len(validation_dataloader)
        early_stopping.check(avg_val_loss)

        print(f"Epoch: {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")
        val_loss_history.append(avg_val_loss)

    # Check for early stopping.
    early_stopping.check(avg_val_loss)
    if not early_stopping.rewind:
        print("Saving early stopping model.")
        torch.save(model.state_dict(), "fcnn_model_early_stopping.pth")
        print("Done saving early stopping model.")
    if early_stopping.early_stop:
        print(f"Triggered early stopping on epoch {epoch + 1}.")
        return  # Exit the training function instead of using 'break'
    
    
    # Save the model and loss.
    print("Saving model and loss.")
    torch.save(model.state_dict(), "fcnn_model_raw.pth")
    np.save("loss_history.npy", np.array(loss_history), allow_pickle=True)
    np.save("val_loss_history.npy", np.array(val_loss_history), allow_pickle=True)
    print("Done saving model and loss.")


# Initialize the model
model = FCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Prepare data loaders
# Convert train data to tensors
x_train_tensor = torch.tensor(x_train_clas.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_clas.to_numpy(), dtype=torch.long)

# Create a TensorDataset for training data
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Create DataLoader for training data
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Since there is no validation data, use the test data for validation
x_test_tensor = torch.tensor(x_test_clas.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_clas.to_numpy(), dtype=torch.long)

validation_dataset = TensorDataset(x_test_tensor, y_test_tensor)
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Train the model
train(
    model=model,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=200,
    device=device
)

# Evaluate the model on the test set
model.eval()

x_test_tensor = torch.tensor(x_test_clas.to_numpy(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_clas.to_numpy(), dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor) * 100

    print(f'Test Accuracy: {accuracy:.3f}%')

