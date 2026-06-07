import glob
import os
import time

import pygame
from sb3_contrib import MaskablePPO

from tetris_wrapper import TetrisEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

STEP_DELAY = 0.3    # seconds between each piece placement — adjust to taste


def get_step_count(filepath):
    filename = os.path.basename(filepath)
    return int(filename.split("_")[1])


def play_game(model, env):
    obs, _ = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True       # skip to next checkpoint
                if event.key == pygame.K_UP:
                    return "faster"   # speed up
                if event.key == pygame.K_DOWN:
                    return "slower"   # slow down

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        time.sleep(STEP_DELAY)

    time.sleep(2)
    return True


def main():
    global STEP_DELAY

    checkpoints = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "tetris_*.zip")),
        key=get_step_count
    )

    if not checkpoints:
        print("No checkpoints found in:", CHECKPOINT_DIR)
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print("SPACE = skip to next | UP = faster | DOWN = slower | close = quit\n")

    env = TetrisEnv(render_mode="yes")

    for checkpoint_path in checkpoints:
        steps = get_step_count(checkpoint_path)
        print(f"Playing: {steps:,} steps  (delay={STEP_DELAY:.2f}s)")
        pygame.display.set_caption(f"Tetris AI — Step {steps:,}")

        model = MaskablePPO.load(checkpoint_path, env=env)
        result = play_game(model, env)

        if result == "faster":
            STEP_DELAY = max(0.05, STEP_DELAY - 0.1)
        elif result == "slower":
            STEP_DELAY = min(1.0, STEP_DELAY + 0.1)
        elif not result:
            break

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
