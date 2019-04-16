import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, latent_dims=100):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(64, 256, kernel_size=5),
            nn.ELU()
        )

        self.fc_mu = nn.Linear(256, self.latent_dims)
        self.fc_logvar = nn.Linear(256, self.latent_dims)

        self.fc_decode = nn.Sequential(nn.Linear(self.latent_dims, 256),
                                       nn.ELU())
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=4),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=2)
        )

    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn(mu.shape)
        return mu + e * std

    def forward(self, input):
        hidden = self.encoder(input)
        mu = self.fc_mu(hidden.squeeze())
        logvar = self.fc_logvar(hidden.squeeze())
        z = self.reparameterization_trick(mu, logvar)
        decoding = self.fc_decode(z)
        decoding = decoding.reshape(decoding.shape[0], decoding.shape[1], 1, 1)
        recon_output = self.decoder(decoding)
        return recon_output, mu, logvar

    # Reconstruction(binary cross entropy loss) + KL divergence losses summed over all elements and batch
    def ELBO(self, recon_x, x, mu, logvar):
        """ 
            return: -ELBO(= BCE + DKL)
        """
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        DKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + DKL
