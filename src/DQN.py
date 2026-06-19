import random
from multiprocessing import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch._prims_common import dtype_or_default

import config
from PER import PrioritizedReplayBuffer


class ValueNetwork(nn.Module):
    def __init__(self, feature_dim, hidden=config.DQN_HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class DQNAgent:
    def __init__(self, feature_dim, device, lr=1e-4, gamma=0.99, buffer_size=100_00, alpha=0.6,max_candidates=config.MAX_PLACEMENTS):
        self.gamma=gamma
        self.device=device
        self.feature_dim=feature_dim
        self.max_candidates = max_candidates
        
        # creating the networks
        self.v_net = ValueNetwork(feature_dim).to(device)
        self.v_target = ValueNetwork(feature_dim).to(device)
        # copying weights
        self.v_target.load_state_dict(self.v_net.state_dict())
        # freeze target network
        self.v_target.eval()

        self.optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)

    def pad_candidates(self, candidates):
        feats = np.zeros((self.max_candidates, self.feature_dim), dtype=np.float32)
        mask = np.zeros(self.max_candidates, dtype=bool)

        n = min(len(candidates), self.max_candidates)
        
        for i in range(n):
            feats[i] = candidates[i]["features"]
            mask[i] = True
        
        return feats, mask

    def select_candidate(self, candidates, epsilon):
        if len(candidates) == 0:
            return None

        # epsilon greedy
        if random.random() < epsilon:
            return random.randrange(len(candidates)) 

        # features
        feats = np.stack([c["features"] for c in candidates]).astype(np.float32)
        rewards = np.array([c["reward"] for c in candidates], dtype=np.float32)
        # whether game is over or not
        dones = np.array([c["done"] for c in candidates], dtype=np.float32)

        # disable gradients
        with torch.no_grad():
            feats_t = torch.as_tensor(feats, dtype=torch.float32,device=self.device).unsqueeze(0)
            values = self.v_net(feats_t).cpu().numpy()

        # calculating Q
        scores = rewards + self.gamma *values*(1.0-dones)

        return int(np.argmax(scores))

    def train_step(self, batch_size, beta):
        # need enough experiences
        if len(self.buffer) < batch_size:
            return None

        # sample batch
        (features, rewards, next_features, next_masks, dones, idxs, is_weights) = self.buffer.sample(batch_size, beta)
        
        features_t = torch.as_tensor(features, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        is_weights_t = torch.as_tensor(is_weights, device=self.device)

        # value of the actual afterstate
        v_pred = self.v_net(features_t)

        # reshaping and masking the next states
        b, max_candidates, feat_dim = next_features.shape
        next_features_t = torch.as_tensor(next_features, dtype=torch.float32, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        flat_next_features = next_features_t.view(-1, feat_dim)

        with torch.no_grad():
            # network choose action
            next_v_online = self.v_net(flat_next_features).view(b,max_candidates)
            # mask invalid action
            next_v_online = next_v_online.masked_fill(~next_masks_t, float("-inf"))            # choose best action
            best_next_idx = next_v_online.argmax(dim=1)

            # target network evaluates
            next_v_target = self.v_target(flat_next_features).view(b, max_candidates)
            next_v_best = next_v_target.gather(1, best_next_idx.unsqueeze(1)).squeeze(1)

            has_valid_next = next_masks_t.any(dim=1).float()
            next_v_best = next_v_best * has_valid_next

            # bellman target
            target = rewards_t + self.gamma * next_v_best * (1.0 - dones_t)

        td_errors = target - v_pred 
        
        # huber loss
        elementwise_loss = F.smooth_l1_loss(v_pred, target, reduction="none")
        loss = (is_weights_t * elementwise_loss).mean()

        # backpropogation
        self.optimizer.zero_grad() # clear gradient
        loss.backward() # compute gradient
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=10.0) # gradient clipping
        self.optimizer.step() # change weights

        self.buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        return loss.item()

    # copy online network into target network
    def update_target(self):
        self.v_target.load_state_dict(self.v_net.state_dict())

# for epsilon
def linear_schedule(start, end, decay_steps, step):
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)

