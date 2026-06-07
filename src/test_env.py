import random
import time

import pygame

from tetris_wrapper import TetrisEnv

env = TetrisEnv(render_mode="yes")

NUM_GAMES = 10
MOVE_DELAY = 0.15 

for game_num in range(NUM_GAMES):
    observation, _ = env.reset()

    total_reward = 0
    steps = 0
    done = False

    print(f"\nStarting Game {game_num + 1}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        
        placements = env.get_valid_placement()

        if len(placements) == 0:
            print("No valid placements!")
            done = True
            break

        action = random.randrange(len(placements))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        env.render()

        time.sleep(MOVE_DELAY)

        done = terminated or truncated

    print(
        f"Game {game_num + 1}: "
        f"Score={env.game.score}, "
        f"Lines={env.game.lines}, "
        f"Reward={total_reward:.2f}, "
        f"Steps={steps}"
    )
    print(obs.shape)
    print(len(obs))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
