import torch
from sumtree import SumTree
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5), alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.tree = SumTree(max_size)

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.size = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        idx = self.tree.data_pointer

        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.done[idx] = done

        # New transitions receive maximum priority
        self.tree.add(self.max_priority**self.alpha)
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=0.4):
        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)

        # --- PREVIOUS IMPLEMENTATION (Commented out) ---
        # weights = np.zeros((batch_size, 1), dtype=np.float32)
        # segment = self.tree.total_priority / batch_size
        # valid_leaves = self.tree.tree[
        #     self.tree.capacity - 1 : self.tree.capacity - 1 + self.size
        # ]
        # total_p = max(float(self.tree.total_priority), 1e-8)
        # min_prob = max(float(np.min(valid_leaves)) / total_p, 1e-8)
        # max_weight = max((min_prob * self.size) ** (-beta), 1e-8)
        # for i in range(batch_size):
        #     a = segment * i
        #     b = segment * (i + 1)
        #     v = np.random.uniform(a, b)
        #     tree_idx, priority, data_idx = self.tree.get_leaf(v)
        #     indices[i] = data_idx
        #     tree_indices[i] = tree_idx
        #     sampling_prob = max(float(priority) / total_p, 1e-8)
        #     weights[i, 0] = ((sampling_prob * self.size) ** (-beta)) / max_weight
        # -----------------------------------------------

        # --- SCHAUL ET AL. (2016) PAPER IMPLEMENTATION (Algorithm 1 & Sec 3.4 / B.2.1) ---
        raw_weights = np.zeros((batch_size, 1), dtype=np.float32)
        segment = self.tree.total_priority / batch_size
        total_p = max(float(self.tree.total_priority), 1e-8)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.get_leaf(v)

            indices[i] = data_idx
            tree_indices[i] = tree_idx

            sampling_prob = max(float(priority) / total_p, 1e-8)
            raw_weights[i, 0] = (sampling_prob * self.size) ** (-beta)

        # Normalize weights so that max_i w_i = 1 (Schaul et al. 2016, Sec 3.4 / App. B.2.1)
        max_w = max(float(np.max(raw_weights)), 1e-8)
        weights = raw_weights / max_w
        weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)

        return (
            torch.FloatTensor(self.state[indices]),
            torch.FloatTensor(self.action[indices]),
            torch.FloatTensor(self.reward[indices]),
            torch.FloatTensor(self.next_state[indices]),
            torch.FloatTensor(self.done[indices]),
            tree_indices,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, tree_indices, td_errors, epsilon=1e-5):
        for idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
