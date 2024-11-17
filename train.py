import torch
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from typing import Callable

from hanabi_learning_environment import pyhanabi

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

register(
    id='hanabi-v0',
    entry_point='envs:HanabiEnvWrapper',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'policy_network': 'MlpPolicy',
    'save_path': 'models/test'
}
train_config = {
    'num_train_envs': 1,
    'n_steps': 100000,
    'n_steps_testing': 640,
    'batch_size': 128, 
    'n_epochs': 5,
}
env_config = {
    "colors":                   5,
    "ranks":                    5,
    "players":                  2,
    "hand_size":                5,
    "max_information_tokens":   8,
    "max_life_tokens":          3,
    "observation_type":         pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
}

def make_env():
    env = gym.make('hanabi-v0', config=env_config)
    return env

def train(model):
    model.learn(
        total_timesteps=train_config['n_steps'],
        reset_num_timesteps=False
    )
    model.save(f"{model_config['save_path']}")

if __name__ == "__main__":
    train_env = DummyVecEnv([make_env for _ in range(train_config['num_train_envs'])])

    model = model_config['algorithm'](
        model_config['policy_network'],
        train_env,
        verbose=2,
        batch_size=train_config['batch_size']
    )
    train(model)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=False)
    print(f"Mean reward: {mean_reward}, std_reward: {std_reward}")
