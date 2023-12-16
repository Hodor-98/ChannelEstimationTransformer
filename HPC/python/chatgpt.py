import torch
import torch.nn as nn
import torch.optim as optim
from models.model import Informer, InformerStack, LSTM,RNN,GRU, InformerStack_e2e

# Assuming you have your training data and other parameters ready
# Initialize your GRU model
input_size = 10  # Replace with your input size
hidden_size = 20  # Replace with your hidden size
num_layers = 2  # Replace with your desired number of layers
features = 5  # Replace with your number of features

gru_model = GRU(features, input_size, hidden_size, num_layers)

# Define your loss function and optimizer
criterion = nn.MSELoss()  # You can use different loss functions based on your task
optimizer = optim.Adam(gru_model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Assuming you have your training data loaded into train_loader
# Iterate through your data for a number of epochs
num_epochs = 10  # Replace with the number of epochs you want to train for

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
gru_model.to(device)  # Move model to the device (CPU or GPU)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs in train_loader:  # Assuming train_loader is your DataLoader
        inputs = inputs.to(device)  # Move input data to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = gru_model.train_data(inputs, device)

        # Calculate the loss
        # Assuming you have your target data in target variable
        loss = criterion(outputs, targets)  # Calculate loss between model output and target

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Finished Training")
