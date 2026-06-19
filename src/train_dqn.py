from collections import deque

import numpy as np
import torch

import config
from DQN import DQNAgent, linear_schedule
from tetris_wrapper import TetrisEnv


def train(num_steps=config.DQN_NUM_STEPS,
          buffer_size=config.DQN_BUFFER_SIZE,
          batch_size=config.DQN_BATCH_SIZE,
          gamma=config.DQN_GAMMA,
          lr=config.DQN_LR,
          train_freq=config.DQN_TRAIN_FREQ,
          target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
          warmup_steps=config.DQN_WARMUP_STEPS,
          epsilon_start=config.DQN_EPSILON_START,
          epsilon_end=config.DQN_EPSILON_END,
          epsilon_decay_steps=config.DQN_EPSILON_DECAY_STEPS,
          beta_start=config.DQN_BETA_START,
          beta_end=config.DQN_BETA_END,
          beta_anneal_steps=config.DQN_BETA_ANNEAL_STEPS,
          alpha=config.DQN_ALPHA,
          checkpoint_path=config.DQN_CHECKPOINT_PATH,
          log_every=config.DQN_LOG_EVERY):
 
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    env = TetrisEnv(render_mode=None)
     
    agent = DQNAgent(config.AFTERSTATE_DIM, device, lr=lr, gamma=gamma,
                      buffer_size=buffer_size, alpha=alpha,
                      max_candidates=config.MAX_PLACEMENTS)
 
    candidates = env.reset() 
 
    episode_reward = 0.0 # reward for current game
    episode_count = 0 # number of completed games
    recent_rewards = deque(maxlen=50) # rewards from last 50 games
    recent_lines = deque(maxlen=50) # lines cleared in last 50 games
 
    # training loop
    for step in range(1, num_steps + 1):
        epsilon = linear_schedule(epsilon_start, epsilon_end,
                                   epsilon_decay_steps, step)
        beta = linear_schedule(beta_start, beta_end,
                                beta_anneal_steps, step)
 
        action_idx = agent.select_candidate(candidates, epsilon)
        chosen = candidates[action_idx]
 
        # actually play the chosen placement
        env.apply_placement(chosen["rotation"], chosen["x_pos"])
 
        reward = chosen["reward"]
        done = chosen["done"]
        features = chosen["features"]
 
        next_candidates = [] if done else env.get_candidates()
        next_features_padded, next_mask = agent.pad_candidates(next_candidates)
 
        agent.buffer.push(features, reward, next_features_padded, next_mask, float(done))
 
        episode_reward += reward

        # waiting for replay buffer to be filled, else collect experience not learn
        if step > warmup_steps and step % train_freq == 0:
            agent.train_step(batch_size, beta)
 
        if step % target_update_freq == 0:
            agent.update_target()

        # game over
        if done:
            episode_count += 1
            recent_rewards.append(episode_reward)
            recent_lines.append(env.game.lines)
            episode_reward = 0.0
            candidates = env.reset()
        else:
            candidates=next_candidates

        # logging stuff
        if step % log_every == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_lines = np.mean(recent_lines) if recent_lines else 0.0
            print(f"step {step} | episodes {episode_count} | "
                  f"epsilon {epsilon:.3f} | beta {beta:.3f} | "
                  f"avg_reward(last50) {avg_reward:.2f} | "
                  f"avg_lines(last50) {avg_lines:.1f} | "
                  f"buffer {len(agent.buffer)}")
            torch.save(agent.v_net.state_dict(), checkpoint_path)
 
    torch.save(agent.v_net.state_dict(), checkpoint_path)
    print("training complete, final model saved to", checkpoint_path)
 
 
if __name__ == "__main__":
    train()
