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
        self.step_cnt = 0

        self.last_obs = {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        obs = super().reset()
        self.last_obs = obs['player_observations'][obs['current_player']]
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        self.step_cnt = 0
        return obs, {}

    def step(self, action):
        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        fireworks = self.last_obs['fireworks']
        prev_life = self.last_obs['life_tokens']

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
        '''
        if prev_life > life:
            reward -= 0.1
        '''
        move_dict = legal_moves[legal_moves_int.index(move)].to_dict()
        if move_dict['action_type'] == "REVEAL_COLOR" or move_dict['action_type'] == "REVEAL_RANK":
            reward -= 0.02
        elif move_dict['action_type'] == "PLAY":
            reward += 0.02
        else:
            reward -= 0.02
        '''
        self.last_obs = obs['player_observations'][obs['current_player']]
        for key, val in self.last_obs['fireworks'].items():
            if fireworks[key] < val and val > 1:
                reward += val * val
        '''
        
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
        truncate = False
        self.step_cnt += 1
        return obs, reward, done, truncate, info
