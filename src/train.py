import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from tetris_wrapper import TetrisEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR = os.path.dirname(BASE_DIR)                   
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
MODEL_DIR      = os.path.join(ROOT_DIR, "models")
LOG_DIR        = os.path.join(ROOT_DIR, "logs")

def mask_fn(env):
    return env.action_masks()

def make_env():
    env = TetrisEnv(render_mode=None)
    env = ActionMasker(env,mask_fn)
    return env

def main():
    vec_env = make_vec_env(make_env,n_envs=6)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1024,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.1,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=15000,
        save_path=CHECKPOINT_DIR,
        name_prefix="tetris"
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=checkpoint_callback
    )

    model.save(os.path.join(MODEL_DIR,"tetris_ppo"))

if __name__ == "__main__":
    main()
