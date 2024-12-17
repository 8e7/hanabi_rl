# https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/master/my_gym/envs/hanabi_env.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from hanabi_learning_environment.rl_env import HanabiEnv
import numpy as np
import os
import torch
from sad_code.load_model import *
import ipdb

class HanabiHumanEvalEnv(HanabiEnv, gym.Env):
    '''
    Eval the ego agent against an agent from SAD or OP.

    config keys:
        'model_path': ego agent's path (sb3 model)
        'other_paths': list of other agent's path (SAD/OP model)
        'other_types': list of other agent's type (either 'sad' or 'op')
        'device': either 'cpu' or 'cuda:x'
    '''
    metadata = {'render_modes': ['human']}
    def __init__(self, config):
        self.config = config
        self.extra_dim = 128
        self.device = config['device']
        super(HanabiHumanEvalEnv, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_shape[0] + 55 + self.extra_dim,), dtype=np.float64)
        self.action_space = spaces.Discrete(self.game.max_moves())
        self.action_num = self.game.max_moves()
        self.seed()
        self.other_agents = []
        for other_path, other_type in zip(config['other_paths'], config['other_types']):
            if other_type == 'sad':
                self.other_agents.append(load_sad_model(other_path, config['device']))
            elif other_type == 'op':
                idx = int(other_path[other_path.find('M')+1:other_path.find('.')])
                self.other_agents.append(load_op_model(other_path, idx, config['device']))
            elif other_type == 'human':
                self.other_agents.append('')
            else:
                print("Invalid model type")
        self.agents = len(self.other_agents)

        if config['baseline'] == False:
            # load policy embeddings
            self.policy_embeddings = []
            self.average_embeddings = []
            for path in config['embedding_paths']:
                policy = torch.load(path)
                padding_dim = self.extra_dim - policy.shape[1] 
                policy = torch.cat([policy.cpu(), torch.zeros(1000, padding_dim).cpu()], dim=1)
                average = policy.mean(dim=0)
                average = average / torch.norm(average)
                self.average_embeddings.append(average)
                self.policy_embeddings.append(policy) 

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def augment_obs(self, obs):
        '''Add policy embeddings to the vectorized observation. Currently just appends zeros.
        '''
        if self.config['baseline']:
            return np.append(obs, np.zeros(self.extra_dim))
        else:
            return np.append(obs, self.other_policy_embedding)

    def other_move(self, obs):
        '''Let the other agent take one step.
        '''
        action_type = input('action_type: ').upper()
        card_index = None
        color = None
        rank = None
        target_offset = None

        if action_type == 'PLAY':
            card_index = int(input('card_index: '))
        elif action_type == 'DISCARD':
            card_index = int(input('card_index: '))
        elif action_type == 'REVEAL_COLOR':
            color = input('color: ').upper()    
            target_offset = int(input('target_offset: '))     
        elif action_type == 'REVEAL_RANK':
            rank = input('rank: ')
            target_offset = int(input('target_offset: '))     

        action = {k: v for k, v in {
            'action_type': action_type,
            'card_index': card_index,
            'color': color,
            'rank': rank,
            'target_offset': target_offset
        }.items() if v is not None}

        obs, reward, done, info = super().step(action)
        vectorized_obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        truncate = False
        return vectorized_obs, reward, done, truncate, info, obs

    def reset(self, seed=None, options=None):
        '''Resets episode.
        options['start_player'] is 1 if the other agent starts first
        '''
        self.seed(seed=seed)
        
        # Decide a new random agent to train against
        self.other_agent_index = self.np_random.integers(0, self.agents)
        self.other_agent = self.other_agents[self.other_agent_index]
        self.other_hid = self.other_agent.get_h0(1)
        self.other_policy_embedding = self.average_embeddings[self.other_agent_index]

        obs = super().reset()
        vectorized_obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        if options is not None and options['start_player'] == 1:
            vectorized_obs, reward, done, truncate, info, obs = self.other_move(vectorized_obs)
        return self.augment_obs(vectorized_obs), {}, obs

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

        vectorized_obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        truncate = False

        # Move opponent
        if done:
            return vectorized_obs, reward, done, truncate, info, obs
        obs, reward, done, truncate, info_op = self.other_move(vectorized_obs)
        vectorized_obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        return self.augment_obs(vectorized_obs), reward, done, truncate, info, obs
