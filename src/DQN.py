import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config
from PER import PrioritizedReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=config.DQN_HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, device, lr=1e-4, gamma=0.99, buffer_size=100_00, alpha=0.6):
        self.n_actions=n_actions
        self.gamma=gamma
        self.device=device
        
        # creating the networks
        self.q_net = QNetwork(obs_dim,n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        # copying weights
        self.q_target.load_state_dict(self.q_net.state_dict())
        # freeze target network
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)

    def select_action(self, obs, mask, epsilon):
        valid_actions = np.flatnonzero(mask)
        if len(valid_actions) == 0:
            return 0

        # epsilon greedy
        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        # disable gradients
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0).cpu().numpy()

        # action masking
        q_values=np.where(mask,q_values,-np.inf)
        return int(np.argmax(q_values))

    def train_step(self, batch_size, beta):
        # need enough experiences
        if len(self.buffer) < batch_size:
            return None

        # sample batch
        (obs, actions, rewards, next_obs, dones, next_masks, idxs, is_weights) = self.buffer.sample(batch_size, beta)
        
        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        is_weights_t = torch.as_tensor(is_weights, device=self.device)

        # q estimate for the action that was actually taken
        q_values = self.q_net(obs_t)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)


        with torch.no_grad():
            # network choose action
            next_q_online = self.q_net(next_obs_t).cpu().numpy()
            # mask invalid action
            next_q_online = np.where(next_masks, next_q_online, -np.inf)
            # choose best action
            next_actions = np.argmax(next_q_online, axis=1)
            next_actions_t = torch.as_tensor(next_actions, device=self.device)

            # target network evaluates
            next_q_target = self.q_target(next_obs_t)
            next_q_taken = next_q_target.gather(1, next_actions_t.unsqueeze(1)).squeeze(1)

            # bellman target
            target = rewards_t + self.gamma * next_q_taken * (1.0 - dones_t)

        td_errors = target - q_taken
        
        # huber loss
        elementwise_loss = F.smooth_l1_loss(q_taken, target, reduction="none")
        loss = (is_weights_t * elementwise_loss).mean()

        # backpropogation
        self.optimizer.zero_grad() # clear gradient
        loss.backward() # compute gradient
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0) # gradient clipping
        self.optimizer.step() # change weights

        self.buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        return loss.item()

    # copy online network into target network
    def update_target(self):
        self.q_target.load_state_dict(self.q_net.state_dict())

# for epsilon
def linear_schedule(start, end, decay_steps, step):
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)

