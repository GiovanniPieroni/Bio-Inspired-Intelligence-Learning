import numpy as np


class SumTree:
    def __init__(self, capacity):
        """Initialize the SumTree with a given capacity."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority):
        """Add a new priority to the tree."""
        tree_index = self.data_pointer + self.capacity - 1
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_index, priority):
        """Update the priority of a given tree index and propagate the change up the tree."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """Retrieve the leaf index, priority, and data index for a given value v of the segment in the batch size."""
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index

    @property
    def total_priority(self):
        """Return the total priority of the tree."""
        return self.tree[0]
