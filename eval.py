import random
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
    id='hanabi-eval',
    entry_point='envs:HanabiEvalEnv',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'model_path': 'models/test'
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

def evaluate(model, env, eval_num=100):
    tot_score = 0
    for iteration in range(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=random.randint(0, 100000000))
        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
        score = env.state.score()
        tot_score += score
        print(score)
    print(f"Average score: {tot_score / eval_num}")
    return tot_score / eval_num
if __name__ == "__main__":
    env = gym.make('hanabi-eval', config=env_config)

    model = model_config['algorithm'].load(model_config['model_path'])
    evaluate(model, env)
