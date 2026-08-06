import torch
import torch.nn as nn


class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.num_centers = num_centers

        self.centers = nn.Parameter(torch.Tensor(num_centers, in_features))
        # Anisotropic sigmas: un valore di spread per ogni dimensione dell'input
        self.sigmas = nn.Parameter(torch.Tensor(num_centers, in_features))

        # Inizializzazione placeholder, da sovrascrivere con i dati reali
        nn.init.uniform_(self.centers, -1.0, 1.0)
        nn.init.constant_(self.sigmas, 1.0)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        x = x.unsqueeze(1)  # (batch_size, 1, in_features)
        c = self.centers.unsqueeze(0)  # (1, num_centers, in_features)
        s = self.sigmas.unsqueeze(0)  # (1, num_centers, in_features)

        # Distanza al quadrato normalizzata (anisotropica)
        dist_sq = torch.sum(((x - c) / s) ** 2, dim=-1)

        # Attivazione Gaussiana
        return torch.exp(-0.5 * dist_sq)

    @torch.no_grad()
    def init_from_data(self, data_batch):
        """
        Sovrascrive centri e spread usando un batch di dati rappresentativi.
        data_batch deve avere shape (num_centers, in_features).
        """
        self.centers.copy_(data_batch)

        # Imposta le sigma in base alla deviazione standard del batch (con un limite inferiore)
        stds = data_batch.std(dim=0, keepdim=True).expand(self.num_centers, -1)
        self.sigmas.copy_(stds.clamp(min=1e-3))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_centers=64):
        super(Actor, self).__init__()
        self.rbf = RBFLayer(state_dim, num_centers)
        self.fc = nn.Linear(num_centers, action_dim)

    def forward(self, state):
        x = self.rbf(state)
        return torch.tanh(self.fc(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_centers=64):
        super(Critic, self).__init__()
        self.rbf = RBFLayer(state_dim + action_dim, num_centers)
        self.fc = nn.Linear(num_centers, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.rbf(x)
        return self.fc(x)
