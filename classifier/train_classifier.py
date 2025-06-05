# classifier/train_classifier.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.fc(h[:, -1, :]))

def train_model(data, labels, device="cpu"):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
                            torch.tensor(labels, dtype=torch.float32))
    train_len = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = LSTMClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss, val_loss = [], []

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_epoch_loss += loss.item()
            val_loss.append(val_epoch_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_loss[-1]:.4f}, Val Loss={val_loss[-1]:.4f}")

    return model, train_loss, val_loss

def main():
    # Load real data
    X_real = np.load("data/X_real.npy")
    y_real = np.zeros(len(X_real))  # Normal
    X_fail = np.load("data/X_fail.npy")
    y_fail = np.ones(len(X_fail))   # Failure

    # Combine real + fail for baseline
    X_combined = np.concatenate([X_real, X_fail])
    y_combined = np.concatenate([y_real, y_fail])

    print("Training on only real data...")
    model_real, train_loss_real, val_loss_real = train_model(X_combined, y_combined)

    # Augment with synthetic data
    X_fake = np.load("gan_model/saved/generated_samples.npy")
    y_fake = np.ones(len(X_fake))  # synthetic failures

    X_augmented = np.concatenate([X_combined, X_fake])
    y_augmented = np.concatenate([y_combined, y_fake])

    print("Training on real + synthetic data...")
    model_aug, train_loss_aug, val_loss_aug = train_model(X_augmented, y_augmented)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(val_loss_real, label="Real Only", linewidth=2)
    plt.plot(val_loss_aug, label="Real + Synthetic", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Classifier Validation Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/classifier_loss_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
