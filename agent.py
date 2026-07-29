import torch
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, np.array([reward]), next_state, np.array([int(done)]))
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, action_dim, exploration_noise=0.1):
        self.action_dim = action_dim
        self.exploration_noise = exploration_noise

    def select_action(self, state, actor_network, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Actor network predicts the action deterministically
        with torch.no_grad():
            action = actor_network(state_tensor).squeeze(0).numpy()

        # Add exploration noise if specified in input parameters
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

        return action
