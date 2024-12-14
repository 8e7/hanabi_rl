# https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/master/my_gym/envs/hanabi_env.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from hanabi_learning_environment.rl_env import HanabiEnv
import numpy as np
import os
import torch
from sad_code.load_model import *

class HanabiMultiTrainEnv(HanabiEnv, gym.Env):
    '''
    Train the ego agent against an agent from SAD or OP.

    config keys:
        'model_path': ego agent's path (sb3 model)
        'other_paths': list of other agent's path (SAD/OP model)
        'other_types': list of other agent's type (either 'sad' or 'op')
        'device': either 'cpu' or 'cuda:x'
        'baseline': If true, uses baseline method (does nothing on augment obs)
        'embedding_paths': list of other agent's policy embedding path
    '''
    metadata = {'render_modes': ['human']}
    def __init__(self, config):
        self.config = config
        self.extra_dim = 128
        self.device = config['device']
        super(HanabiMultiTrainEnv, self).__init__(config=self.config)

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
                self.average_embeddings.append(average)
                self.policy_embeddings.append(policy) 

        self.start_player = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def augment_obs(self, obs):
        '''Add policy embeddings to the vectorized observation.
        '''
        if self.config['baseline']:
            return np.append(obs, np.zeros(self.extra_dim))
        else:
            return np.append(obs, self.other_policy_embedding)

    def other_move(self, obs):
        '''Let the other agent take one step.
        '''
        legal_moves = np.zeros(self.action_num+1, dtype=int)
        legal_moves_int = [self.game.get_move_uid(move) for move in self.state.legal_moves()]
        legal_moves[legal_moves_int] = 1
        obs_tensor = torch.from_numpy(obs).float().view((1, 838)).to(self.device)
        legal_moves = torch.from_numpy(legal_moves).to(self.device)
        action, new_hid = self.other_agent.greedy_act(obs_tensor, legal_moves,self.other_hid)
        self.other_hid = new_hid
        action = action.item()

        obs, reward, done, info = super().step(action)
        truncate = False
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        return obs, reward, done, truncate, info

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
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        # random start player
        if self.start_player:
            obs, reward, done, truncate, info = self.other_move(obs)
        self.start_player = (self.start_player + 1) % 2
        return self.augment_obs(obs), {}

    def step(self, action):
        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        # the legal move array may be too small in some cases, so just modulo action by the array length
        legal_moves = self.state.legal_moves()
        legal_moves_int = [self.game.get_move_uid(move) for move in legal_moves]
            
        reward = 0
        move = int(action)
        if action not in legal_moves_int:
            move = legal_moves_int[0]
        obs, r, done, info = super().step(move)
        reward += r
        life = obs['player_observations'][obs['current_player']]['life_tokens']
        if done and life > 0:
            reward += 10
        move_dict = legal_moves[legal_moves_int.index(move)].to_dict()
        if move_dict['action_type'] == "REVEAL_COLOR" or move_dict['action_type'] == "REVEAL_RANK":
            reward -= 0.02
        elif move_dict['action_type'] == "PLAY":
            reward += 0.02
        else:
            reward -= 0.02

        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        truncate = False
        # Move opponent
        if done:
            return self.augment_obs(obs), reward, done, truncate, info
        obs, reward_op, done, truncate, info_op = self.other_move(obs)
        reward += reward_op
        return self.augment_obs(obs), reward, done, truncate, info
