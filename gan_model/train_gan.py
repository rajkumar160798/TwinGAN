# gan_model/train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=5, seq_len=50):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        return self.linear(h)

class Discriminator(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.linear(h[:, -1, :]))

def train_gan(data_path="data/X_fail.npy", epochs=100, batch_size=32, device="cpu"):
    data = np.load(data_path)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optim_G = optim.Adam(G.parameters(), lr=0.0002)
    optim_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_batch, in dataloader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, 50, 10).to(device)
            fake_batch = G(z)

            D_real = D(real_batch)
            D_fake = D(fake_batch.detach())
            loss_D = criterion(D_real, torch.ones_like(D_real)) +                      criterion(D_fake, torch.zeros_like(D_fake))

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Train Generator
            D_fake = D(fake_batch)
            loss_G = criterion(D_fake, torch.ones_like(D_fake))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Save models
    os.makedirs("gan_model/saved", exist_ok=True)
    torch.save(G.state_dict(), "gan_model/saved/generator.pth")
    torch.save(D.state_dict(), "gan_model/saved/discriminator.pth")

    # Save sample outputs
    with torch.no_grad():
        z = torch.randn(5, 50, 10).to(device)
        samples = G(z).cpu().numpy()
        np.save("gan_model/saved/generated_samples.npy", samples)
        print("Saved sample sequences to gan_model/saved/")

if __name__ == "__main__":
    train_gan()
