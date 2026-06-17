import os
import warnings

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, start_ent=0.05, end_ent=0.01, total_steps=50_000_000):
        super().__init__()
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.total_steps = total_steps

    def _on_step(self):
        progress = self.num_timesteps / self.total_steps
        self.model.ent_coef = max(self.end_ent, self.start_ent * (1.0 - progress))
        return True

def main():
    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    RESUME_FROM = os.path.join(CHECKPOINT_DIR,"tetris_2000000_steps.zip")

    
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=2048,
        n_epochs=4,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.05,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[512, 512, 512]),
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
        save_freq=500000,
        save_path=CHECKPOINT_DIR,
        name_prefix="tetris_"
    )

    model.learn(
    total_timesteps=50_000_000,
    callback=[checkpoint_callback, EntropyScheduleCallback()],
    )

    model.save(os.path.join(MODEL_DIR,"tetris_ppo"))
    print("Saved to tetris_ppo")

if __name__ == "__main__":
    main()
