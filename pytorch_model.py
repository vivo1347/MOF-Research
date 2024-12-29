import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define a simple feedforward neural network in PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, activation_fn):
        super(NeuralNetwork, self).__init__()
        layers = []
        in_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation_fn())
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, 1))
        
        # Combine the layers into a sequential container
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def main():
    # Loading from file and initializing
    dev_file = np.load("train_mix.npz", allow_pickle=True)
    X_dev = dev_file["x"]
    Y_dev = dev_file["y"]

    test_file = np.load("test_good.npz", allow_pickle=True)
    X_test = test_file["x"]
    Y_test = test_file["y"]

    # Impute NaN values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_dev = imputer.fit_transform(X_dev)
    X_test = imputer.transform(X_test)
    
    # Standardize the data
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert data to PyTorch tensors
    X_dev_tensor = torch.tensor(X_dev_scaled, dtype=torch.float32)
    Y_dev_tensor = torch.tensor(Y_dev, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Define dataset and dataloader
    train_dataset = TensorDataset(X_dev_tensor, Y_dev_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model parameters
    input_size = X_dev.shape[1]
    hidden_layer_sizes = [100, 50]  # Example architecture
    activation_fn = nn.ReLU

    # Initialize the model
    model = NeuralNetwork(input_size, hidden_layer_sizes, activation_fn)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with gradient clipping
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        
        # Check for NaN in predictions
        if np.isnan(predictions).any():
            raise ValueError("Model predictions contain NaN values. Consider adjusting learning rate or network architecture.")
        
        r2 = r2_score(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        mae = mean_absolute_error(Y_test, predictions)
    
    print("R2 score on the test set:", r2)
    print("MSE on the test set:", mse)
    print("MAE on the test set:", mae)
    
    # Plot parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_test, predictions, edgecolors=(0, 0, 0), alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
