import os
import time
import warnings

import numpy as np
import pygame
import torch

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)
import config
import tetris as tet
from DQN import ValueNetwork
from tetris_wrapper import TetrisEnv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_per_tetris.pt")

STEP_DELAY = 0.05


def select_best_candidate(v_net, candidates, gamma, device):
    feats = np.stack([c["features"] for c in candidates]).astype(np.float32)
    rewards = np.array([c["reward"] for c in candidates], dtype=np.float32)
    dones = np.array([c["done"] for c in candidates], dtype=np.float32)

    with torch.no_grad():
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
        values = v_net(feats_t).cpu().numpy()

    scores = rewards + gamma * values * (1.0 - dones)
    return int(np.argmax(scores))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading model from {MODEL_PATH}")
    env = TetrisEnv(render_mode=None)   

    v_net = ValueNetwork(config.AFTERSTATE_DIM).to(device)
    v_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    v_net.eval()

    gamma = config.DQN_GAMMA

    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    clk    = pygame.time.Clock()

    game_number = 0
    running     = True

    while running:
        game_number += 1
        candidates = env.reset()
        done       = False
        pygame.display.set_caption(f"Tetris AI — Game {game_number}")

        while not done and running and candidates:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            chosen_idx = select_best_candidate(v_net, candidates, gamma, device)
            chosen     = candidates[chosen_idx]

            
            screen.fill(config.BLACK)

            pygame.draw.rect(screen, config.WHITE,
                (config.BOARD_X - 5, config.BOARD_Y - 5,
                 config.PLAYFIELD_WIDTH + 10, config.PLAYFIELD_HEIGHT + 10), 5)
            pygame.draw.rect(screen, config.WHITE,
                (config.NEXT_BOX_X - 5, config.NEXT_BOX_Y, 140, 120), 4)

            tet.draw_next_piece(screen, env.game.next_piece)
            tet.draw(env.game, screen)
            tet.draw_score(screen, env.game.score, env.game.lines,
                           config.SCORE_BOX_X, config.SCORE_BOX_Y)

            if env.game.game_over:
                tet.draw_game_over(screen,
                    config.WIDTH // 2 - 100, config.HEIGHT // 2 - 30)

            pygame.display.flip()
            clk.tick(60)
            time.sleep(STEP_DELAY)

            env.apply_placement(chosen["rotation"], chosen["x_pos"])
            done = chosen["done"]
            candidates = [] if done else env.get_candidates()

        if done:
            print(f"Game {game_number:3d} — Score: {env.game.score:5d} | Lines: {env.game.lines:3d}")
            time.sleep(1.0)

    pygame.quit()
    print(f"Stopped after {game_number} games.")


if __name__ == "__main__":
    main()
