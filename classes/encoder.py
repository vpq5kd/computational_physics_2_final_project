import torch
import torch.nn as nn


class ECGAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )

        self.to_latent = nn.Linear(128 * 125, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * 125)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                64, 32,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 12,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.to_latent(x)
        return x

    def decode(self, z):
        x = self.from_latent(z)
        x = x.view(-1, 128, 125)
        x = self.decoder(x)
        x = x[:, :, :1000]
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
