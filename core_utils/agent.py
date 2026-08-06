import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Agent:

    def __init__(
        self, action_dim, exploration_noise=0.1, min_noise=0.01, noise_decay=0.995
    ):
        self.action_dim = action_dim
        self.exploration_noise = exploration_noise
        self.min_noise = min_noise
        self.noise_decay = noise_decay

    # def select_action(self, state, actor_network, add_noise=False):
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0)

    #     # Keep the action as a Tensor to be able to apply torch functions
    #     with torch.no_grad():
    #         action_tensor = actor_network(state_tensor).squeeze(0)

    #     if add_noise:
    #         noise = torch.randn_like(action_tensor) * self.exploration_noise
    #         action_tensor = torch.clamp(action_tensor + noise, -1.0, 1.0)

    #         self.exploration_noise = max(
    #             self.min_noise, self.exploration_noise * self.noise_decay
    #         )

    #     # Convert to numpy array only at return time
    #     return action_tensor.numpy()

    def select_action(self, state, actor_network, add_noise=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if not torch.isfinite(state_tensor).all():
            raise RuntimeError(
                f"[Agent] Input state contains non-finite values: {state}"
            )

        with torch.no_grad():
            raw_action = actor_network(state_tensor).squeeze(0)

        if not torch.isfinite(raw_action).all():
            bad_params = [
                n
                for n, p in actor_network.named_parameters()
                if not torch.isfinite(p).all()
            ]
            raise RuntimeError(
                f"[Agent] Actor produced non-finite action: {raw_action}\n"
                f"Input state (finite, so not the state's fault): {state}\n"
                f"Actor parameters already non-finite: {bad_params if bad_params else 'NONE -> NaN arises fresh in this forward pass (e.g., RBF collapse)'}"
            )

        action_tensor = raw_action
        if add_noise:
            noise = torch.randn_like(action_tensor) * self.exploration_noise
            action_tensor = torch.clamp(action_tensor + noise, -1.0, 1.0)
            # self.exploration_noise = max(
            #     self.min_noise, self.exploration_noise * self.noise_decay
            # )

        return action_tensor.numpy()
