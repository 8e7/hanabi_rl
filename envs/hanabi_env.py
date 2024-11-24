# https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/master/my_gym/envs/hanabi_env.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from hanabi_learning_environment.rl_env import HanabiEnv
import numpy as np

class HanabiEnvWrapper(HanabiEnv, gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, config):
        self.config = config
        super(HanabiEnvWrapper, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.MultiBinary(observation_shape[0])
        self.action_space = spaces.Discrete(self.game.max_moves())

        self.seed()
        self.illegal_move_reward = -1
        self.invalid_cnt = 0
    
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
        legal_moves = self.state.legal_moves()
        legal_moves_int = [self.game.get_move_uid(move) for move in legal_moves]
            
        reward = 0
        move = int(action)
        if action not in legal_moves_int:
            move = legal_moves_int[0]
        obs, r, done, info = super().step(move)
        reward += r
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])

        truncate = False
        return obs, reward, done, truncate, info
