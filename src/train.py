import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import torch.nn as nn
from .model import model 
import pandas as pd
from sklearn.model_selection import train_test_split
from .load_config import load_constants_from_yaml
constants = load_constants_from_yaml('constants.yml')
batch_size = constants["BATCH_SIZE"]
epochs = constants["EPOCHS"]
validation_split = constants["VALIDATION_SPLIT"]
test_size = constants["TEST_SIZE"]
random_state = constants["RANDOM_STATE"]
processed_data_path = constants["PROCESSED_DATA_PATH"]


def train_model(X_train, y_train, batch_size, epochs, validation_split):

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create a dataset and dataloaders
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model, criterion, and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            labels = labels.unsqueeze_(-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    print("Training complete.")
    return model

if __name__ == "__main__":
    # load data
    df = pd.read_csv(processed_data_path+"df_transformed.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    train_model(X_train.values, y_train.values, batch_size, epochs, validation_split)