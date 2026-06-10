import logging
import os
import warnings

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)

import config
from tetris_wrapper import TetrisEnv

warnings.filterwarnings("ignore")
logging.getLogger("stable_baselines3").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_STEPS  = 1_000_000
N_ENVS       = 8
N_EVAL_GAMES = 30
N_TRIALS     = 55

def mask_fn(env): return env.action_masks()

def make_env():
    env = TetrisEnv(render_mode=None)
    return ActionMasker(env, mask_fn)

def evaluate(model):
    env = TetrisEnv(render_mode=None)
    lines = []
    for _ in range(N_EVAL_GAMES):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=env.action_masks())
            obs, _, done, _, _ = env.step(action)
        lines.append(env.game.lines)
    return float(np.mean(lines))

def set_config(trial):
    params = {
        'SURVIVAL_BONUS':         trial.suggest_float("survival_bonus",          0.05, 4.0,  log=True),
        'NEW_HOLES_PENALTY':      trial.suggest_float("new_holes_penalty",        0.5,  7.0),
        'EXISTING_HOLES_PENALTY': trial.suggest_float("existing_holes_penalty",   0.005, 0.3, log=True),
        'HEIGHT_PENALTY':         trial.suggest_float("height_penalty",           0.003, 0.1, log=True),
        'BUMPINESS_PENALTY':      trial.suggest_float("bumpiness_penalty",        0.003, 0.15,log=True),
        'GAME_OVER_PENALTY':      trial.suggest_float("game_over_penalty",        5.0,  80.0),
        'NEAR_COMPLETE_BONUS':    trial.suggest_float("near_complete_bonus",      0.0,   2.0),
        'RIGHT_WELL_BONUS':       trial.suggest_float("right_well_bonus",         0.0,   1.0),
        'FAT_TET_BONUS' :         trial.suggest_int("fat_tet_bonus",              100,   300),
    }
    for k, v in params.items():
        setattr(config, k, v)
    return params  

def objective(trial):
    params = set_config(trial)

    def make_env_fn():
        import config as cfg

        for k, v in params.items():
            setattr(cfg, k, v)

        env = TetrisEnv(render_mode=None)
        return ActionMasker(env, mask_fn)

    vec_env = make_vec_env(
        make_env_fn,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    try:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            n_steps=512,
            n_epochs=10,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.999,
            ent_coef=0.05,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
        )

        model.learn(total_timesteps=TRAIN_STEPS)

        score = evaluate(model)

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        score = 0.0

    finally:
        vec_env.close()

    print(f"Trial {trial.number:3d} | lines={score:5.1f} | {trial.params}")

    return score

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="tetris_reward_tuning"
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== BEST ===")
    print(f"Mean lines: {study.best_value:.1f}")
    for k, v in study.best_params.items():
        print(f"  {k} = {v:.6f}")
