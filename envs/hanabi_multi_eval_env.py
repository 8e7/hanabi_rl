# https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/master/my_gym/envs/hanabi_env.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from hanabi_learning_environment.rl_env import HanabiEnv
import numpy as np
import os
import torch
from sad_code.load_model import *

class HanabiMultiEvalEnv(HanabiEnv, gym.Env):
    '''
    Evaluates the ego agent against an agent from SAD or OP.

    config keys:
        'model_path': ego agent's path (sb3 model)
        'other_path': other agent's path (SAD/OP model)
        'other_type': other agent's type (either 'sad' or 'op')
        'device': either 'cpu' or 'cuda:x'
    '''
    metadata = {'render_modes': ['human']}
    def __init__(self, config):
        self.config = config
        super(HanabiMultiEvalEnv, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.MultiBinary(observation_shape[0])
        self.action_space = spaces.Discrete(self.game.max_moves())
        self.action_num = self.game.max_moves()

        self.seed()
        if config['other_type'] == 'sad':
            self.other_agent = load_sad_model(config['other_path'], config['device'])
        elif config['other_type'] == 'op':
            idx = os.path.dirname(config['other_path'])
            self.other_agent = load_op_model(config['other_path'], idx, config['device'])
        else:
            print("Invalid model type")

        self.other_hid = self.other_agent.get_h0(1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def other_move(self, obs):
        '''Let the other agent take one step.
        '''
        legal_moves = np.zeros(self.action_num, dtype=int)
        legal_moves_int = [self.game.get_move_uid(move) for move in self.state.legal_moves()]
        legal_moves[legal_moves_int] = 1
        obs_input = {
            'priv_s': torch.from_numpy(obs),
            'legal_moves': legal_moves,
            'h0': self.other_hid['h0'],
            'c0': self.other_hid['c0']
        }
        reply = self.other_agent.act(obs_input)
        action = reply['a']
        self.other_hid = {'h0': reply['h0'], 'c0': reply['c0']}

        obs, reward, done, info = super().step(action)
        truncate = False
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        return obs, reward, done, truncate, info

    def reset(self, seed=None, options=None):
        '''Resets episode.
        options['start_player'] is 1 if the other agent starts first
        '''
        self.seed(seed=seed)
        obs = super().reset()
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        if options is not None and options['start_player'] == 1:
            obs, reward, done, truncate, info = self.other_move(obs)
        return obs, {}

    def step(self, action):
        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        # the legal move array may be too small in some cases, so just modulo action by the array length
        legal_moves = self.state.legal_moves()
        legal_moves_int = [self.game.get_move_uid(move) for move in legal_moves]
            
        move = int(action)
        if action not in legal_moves_int:
            move = legal_moves_int[0]

        obs, reward, done, info = super().step(move)
        if action not in legal_moves_int:
            info['illegal_move'] = 1

        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        truncate = False

        # Move opponent
        if done:
            return obs, reward, done, truncate, info
        obs, reward, done, truncate, info_op = self.other_move(obs)
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        return obs, reward, done, truncate, info
