import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.005):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)    
        
        # Actor network
        self.actor_mu = nn.Linear(64, action_dim)
        self.actor_sigma = nn.Linear(64, action_dim)
        
        # Critic network
        self.critic = nn.Linear(64, 1)
        
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        
        # Actor
        mu = self.actor_mu(x)
        sigma = torch.exp(self.actor_sigma(x))  # Ensure sigma is positive
        
        # Critic
        value = self.critic(x)
        
        return mu, sigma, value