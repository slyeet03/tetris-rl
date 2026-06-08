import os
import time

import pygame
from sb3_contrib import MaskablePPO

from tetris_wrapper import TetrisEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "tetris_ppo.zip")

STEP_DELAY = 0.3  

def main():
    print(f"Loading model from {MODEL_PATH}")
    env = TetrisEnv(render_mode="yes")
    model = MaskablePPO.load(MODEL_PATH, env=env)

    game_number = 0

    print("Running — close the window to stop")
    print("UP = faster | DOWN = slower\n")

    global STEP_DELAY
    running = True

    while running:
        game_number += 1
        obs, _ = env.reset()
        done = False
        total_reward = 0

        pygame.display.set_caption(f"Tetris AI — Game {game_number}")

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        STEP_DELAY = max(0.05, STEP_DELAY - 0.05)
                        print(f"Speed up — delay={STEP_DELAY:.2f}s")
                    if event.key == pygame.K_DOWN:
                        STEP_DELAY = min(1.0, STEP_DELAY + 0.05)
                        print(f"Slow down — delay={STEP_DELAY:.2f}s")

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            time.sleep(STEP_DELAY)

        if done:
            print(f"Game {game_number} over — Score: {env.game.score} | Lines: {env.game.lines} | Reward: {total_reward:.1f}")
            time.sleep(1.5) 
    pygame.quit()
    print(f"\nStopped after {game_number} games.")


if __name__ == "__main__":
    main()
