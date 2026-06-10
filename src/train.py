import os
import warnings

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)
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
    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    RESUME_FROM = os.path.join(CHECKPOINT_DIR,"tetris_2000000_steps.zip")

    
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        n_epochs=10,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.999,
        ent_coef=0.05,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        verbose=1
    )

    '''
    model = MaskablePPO.load(
        RESUME_FROM,
        env=vec_env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        custom_objects={
            "ent_coef": 0.1,
            "learning_rate": get_linear_fn(3e-4, 1e-5, 1.0),
        }
    )
    '''
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path=CHECKPOINT_DIR,
        name_prefix="tetris_"
    )

    model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_callback,
        #reset_num_timesteps=False
    )

    model.save(os.path.join(MODEL_DIR,"tetris_ppo"))
    print("Saved to tetris_ppo")

if __name__ == "__main__":
    main()
