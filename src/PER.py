import random

import numpy as np

from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.7, eps=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0

    def push(self, obs, action, reward, next_obs, done, next_mask):
        transition = (obs,action,reward,next_obs,done,next_mask)

        # new transition getting max priority so it gets sampled atleast once
        self.tree.add(self.max_priority**self.alpha, transition)

    # using stratified sampling
    def sample(self, batch_size, beta):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i+1)
            sample_val = random.uniform(low, high)
            idx, priority, data = self.tree.get(sample_val)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        is_weights /= is_weights.max()
        
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)

        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
            idxs,
            np.array(is_weights, dtype=np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority=max(self.max_priority,priority)

    def __len__(self):
        return self.tree.n_entries

