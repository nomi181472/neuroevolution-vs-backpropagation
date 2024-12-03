import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from plotGraph import plot_metrics

# Load and Prepare Tabular Dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode target labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Define PyTorch Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# Define the Feedforward Neural Network
class TabularModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(TabularModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.fc(x)

# Training and Evaluation Functions
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)


# Hyperparameter Grid
param_grid = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32],
    "hidden_sizes": [(64, 32), (128, 64)],
}

param_combinations = list(product(*param_grid.values()))
grid_search_metrics = []

print("\nApply GridSearch Serach on learning rate, batch size, hidden state\n")

# Grid Search Training
for params in param_combinations:
    learning_rate, batch_size, hidden_sizes = params

    # Update DataLoader with the current batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model with the current hidden sizes
    model = TabularModel(input_dim=X.shape[1], output_dim=y_onehot.shape[1], hidden_sizes=hidden_sizes)

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics Storage
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    for epoch in range(30):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

    model_state = model.state_dict()

    grid_search_metrics.append({
        "params": params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
        "model_state": model_state,  # Save the model state
    })
    print(f"Finished training for params: {params}")

# Plot Metrics for All Grid Search Configurations
for metrics in grid_search_metrics:
    params = metrics["params"]
    epochs = np.arange(1, len(metrics["train_losses"]) + 1)

    # Plot Loss
    plot_metrics(
        metric_values=[metrics["train_losses"], metrics["val_losses"], metrics["test_losses"]],
        epochs=epochs,
        labels=["Train Loss", "Validation Loss", "Test Loss"],
        title=f"Loss for Params: LR={params[0]}, Batch={params[1]}, Hidden={params[2]}",
        ylabel="Loss",
        linestyle=['-', '--', ':']
    )

    # Plot Accuracy
    plot_metrics(
        metric_values=[metrics["train_accuracies"], metrics["val_accuracies"], metrics["test_accuracies"]],
        epochs=epochs,
        labels=["Train Accuracy", "Validation Accuracy", "Test Accuracy"],
        title=f"Accuracy for Params: LR={params[0]}, Batch={params[1]}, Hidden={params[2]}",
        ylabel="Accuracy",
        linestyle=['-', '--', ':']
    )

# Best Model Selection
best_model_metrics = max(grid_search_metrics, key=lambda x: x["val_accuracies"][-1])
best_params = best_model_metrics["params"]
epochs = np.arange(1, len(best_model_metrics["train_losses"]) + 1)

print("\nBest Model Parameters:")
print(f"Learning Rate: {best_params[0]}, Batch Size: {best_params[1]}, Hidden Sizes: {best_params[2]}")

print("\nEpoch-wise Metrics for Best Model:")
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
for epoch in range(len(train_losses)):
    print(f"{epoch + 1:<6} {train_losses[epoch]:<12.4f} {val_losses[epoch]:<12.4f} {test_losses[epoch]:<12.4f} "
          f"{train_accuracies[epoch]:<12.4f} {val_accuracies[epoch]:<12.4f} {test_accuracies[epoch]:<12.4f}")


# Plot Metrics for Best Model
plot_metrics(
    metric_values=[best_model_metrics["train_losses"], best_model_metrics["val_losses"], best_model_metrics["test_losses"]],
    epochs=epochs,
    labels=["Train Loss", "Validation Loss", "Test Loss"],
    title=f"Loss for Best Model: LR={best_params[0]}, Batch={best_params[1]}, Hidden={best_params[2]}",
    ylabel="Loss",
    linestyle=['-', '--', ':']
)

plot_metrics(
    metric_values=[best_model_metrics["train_accuracies"], best_model_metrics["val_accuracies"], best_model_metrics["test_accuracies"]],
    epochs=epochs,
    labels=["Train Accuracy", "Validation Accuracy", "Test Accuracy"],
    title=f"Accuracy for Best Model: LR={best_params[0]}, Batch={best_params[1]}, Hidden={best_params[2]}",
    ylabel="Accuracy",
    linestyle=['-', '--', ':']
)

import csv
metrics_file = "supervisedDL-performance.csv"
with open(metrics_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Test Loss", "Train Acc", "Val Acc", "Test Acc"])
    for epoch in range(len(best_model_metrics["train_losses"])):
        writer.writerow([
            epoch + 1,
            best_model_metrics["train_losses"][epoch],
            best_model_metrics["val_losses"][epoch],
            best_model_metrics["test_losses"][epoch],
            best_model_metrics["train_accuracies"][epoch],
            best_model_metrics["val_accuracies"][epoch],
            best_model_metrics["test_accuracies"][epoch]
        ])
print(f"Metrics saved to {metrics_file}")


# Reload the best model with its parameters
input_dim = X.shape[1]
output_dim = y_onehot.shape[1]
best_model = TabularModel(input_dim=input_dim, output_dim=output_dim, hidden_sizes=[64, 32])
best_model.load_state_dict(best_model_metrics["model_state"])  # Load the best model's state_dict
best_model.eval()  # Set the model to evaluation mode

# Define the test_and_compare function
def test_and_compare(model, dataloader, num_examples=5):
    """
    Test the model and compare predictions with actual values.

    Parameters:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the dataset.
        num_examples: Maximum number of examples to display.
    """
    model.eval()
    examples_displayed = 0

    with torch.no_grad():
        print("\n--- Sample Predictions ---\n")
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            actuals = y_batch.argmax(dim=1).cpu().numpy()
            features = X_batch.cpu().numpy()

            # Iterate over batch examples
            for idx in range(len(features)):
                print(f"Features: {features[idx]}")
                print(f"Predicted: {predictions[idx]}, Actual: {actuals[idx]}")
                print("-" * 50)
                examples_displayed += 1

                # Stop after displaying the specified number of examples
                if examples_displayed >= num_examples:
                    return

# Run the comparison on the training set
print("\n--- Training Data Predictions (Best Model) ---")
test_and_compare(best_model, train_loader, num_examples=5)

# Run the comparison on the test set
print("\n--- Test Data Predictions (Best Model) ---")
test_and_compare(best_model, test_loader, num_examples=5)