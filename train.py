import torch
from torch import cuda
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from typing import Callable

from hanabi_learning_environment import pyhanabi

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from eval import evaluate

register(
    id='hanabi-v0',
    entry_point='envs:HanabiEnvWrapper',
)
register(
    id='hanabi-eval',
    entry_point='envs:HanabiEvalEnv',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'policy_network': 'MlpPolicy',
    'save_path': 'models/PPO_fix'
}
train_config = {
    'num_train_envs': 4,
    'n_steps': 50000,
    'n_steps_testing': 640,
    'batch_size': 128, 
    'n_epochs': 100,
    'learning_rate': 1e-4
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

def train(model, eval_env):
    best_avg = 0
    for epoch in range(train_config['n_epochs']):
        print("Epoch", epoch)
        model.learn(
            total_timesteps=train_config['n_steps'],
            reset_num_timesteps=False
        )
        avg_score = evaluate(model, eval_env, eval_num=10)
        if avg_score > best_avg:
            best_avg = avg_score
            print("Saving Model")
            model.save(f"{model_config['save_path']}")

if __name__ == "__main__":
    train_env = SubprocVecEnv([make_env for _ in range(train_config['num_train_envs'])])
    eval_env = gym.make('hanabi-eval', config=env_config)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU, net_arch=[256, 256] 
    )
    model = model_config['algorithm'](
        model_config['policy_network'],
        train_env,
        verbose=2,
        batch_size=train_config['batch_size'],
        learning_rate=train_config['learning_rate'],
        policy_kwargs=policy_kwargs
    )
   
    print(model.policy)
    train(model, eval_env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=False)
    print(f"Mean reward: {mean_reward}, std_reward: {std_reward}")
