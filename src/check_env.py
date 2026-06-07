# check_env.py

from gymnasium.utils.env_checker import check_env

from tetris_wrapper import TetrisEnv

env = TetrisEnv()

check_env(env)

print("Environment passed!")
