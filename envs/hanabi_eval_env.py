# https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/master/my_gym/envs/hanabi_env.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from hanabi_learning_environment.rl_env import HanabiEnv
import numpy as np

class HanabiEvalEnv(HanabiEnv, gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, config):
        self.config = config
        super(HanabiEvalEnv, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.MultiBinary(observation_shape[0])
        self.action_space = spaces.Discrete(self.game.max_moves())

        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        obs = super().reset()
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        return obs, {}

    def step(self, action):
        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        # the legal move array may be too small in some cases, so just modulo action by the array length
        legal_moves = self.state.legal_moves()
        move = legal_moves[action % len(legal_moves)].to_dict()

        obs, reward, done, info = super().step(move)
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])

        truncate = False
        return obs, reward, done, truncate, info
