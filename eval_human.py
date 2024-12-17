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
from sb3_contrib import RecurrentPPO

from utils import bcolors, read_agents_file
import numpy as np
import ipdb

register(
    id='hanabi-eval-human',
    entry_point='envs:HanabiHumanEvalEnv',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'model_path': 'models/PPO_CLIP_average_norm.zip',
}
env_config = {
    "other_paths": ['sad_models/sad_models/sad_2p_3.pthw', 'sad_models/sad_models/sad_2p_4.pthw'],
    "other_types": ['sad', 'sad'],
    'device': 'cpu',
    "colors":                   5,
    "ranks":                    5,
    "players":                  2,
    "hand_size":                5,
    "max_information_tokens":   8,
    "max_life_tokens":          3,
    "observation_type":         pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
    "baseline": False,
    "embedding_paths": [f'agent_embeddings/{i}.pt' for i in range(40)],
}

game_setting = [
    [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'Y', 'rank': 0}, {'color': 'Y', 'rank': 0}, {'color': 'R', 'rank': 1}, {'color': 'G', 'rank': 3}, {'color': 'B', 'rank': 1}]],
    [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'R', 'rank': 0}, {'color': 'W', 'rank': 2}, {'color': 'R', 'rank': 2}, {'color': 'R', 'rank': 0}, {'color': 'Y', 'rank': 1}]]
]

def evaluate(model, env, eval_num=100, vis=False):
    tot_score = 0
    tot_step = 0
    illegal_step = 0
    scores = {}
    for iteration in range(eval_num):
        done = False
        vectorized_obs, info, obs = env.reset(seed=random.randint(0, 100000000), options={'start_player': iteration % 2})
        #lstm_states = None
        #episode_starts = 1
        while not done:
            action, _state = model.predict(vectorized_obs, deterministic=True)
            print(action)
            vectorized_obs, reward, done, _, info, obs = env.step(action)
            if 'illegal_move' in info:
                illegal_step += 1
            tot_step += 1
            #action, lstm_states = model_list[player].predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            #obs, reward, done, _, info = env.step(action)
            #episode_starts = done
            
        score = env.unwrapped.state.score()
        tot_score += score
        agent_ind = env.unwrapped.other_agent_index
        if agent_ind not in scores:
            scores[agent_ind] = []
        scores[agent_ind].append(score)

    print(f"Average score: {tot_score / eval_num}")
    print(f"Illegal step rate: {illegal_step / tot_step}")
    for agent_ind, score in scores.items():
        print(f"Agent {agent_ind}: {np.mean(score)}")
    return tot_score / eval_num


if __name__ == "__main__":
    agents_path, agents_type = read_agents_file('training_agents.txt')
    env_config['other_paths'] = agents_path
    env_config['other_types'] = agents_type

    env = gym.make('hanabi-eval-human', config=env_config)
    model = model_config['algorithm'].load(model_config['model_path'])
    evaluate(model, env, eval_num=1)
