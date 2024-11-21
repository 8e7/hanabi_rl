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
        self.record_episodes = self.config['record_episodes']
        if self.record_episodes:
            self.episode_file = open(self.config['episodes_file'],"a+")
        else:
            self.episode_file = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def obs_vec_compress(self, vec):
        # save 0,1 vec as hex string and filled with leading zeros
        # two player game: len(vec) = 658
        str_len = (len(vec)+3) // 4
        return hex(int("".join(str(x) for x in vec), 2))[2:].zfill(str_len)

    def pyhanabi_compress(self, s):
        return s.replace("\n", ",")

    def write_obs(self, obs):
        self.episode_file.write(str(obs['current_player']) + "\n")
        for i in range(len(obs['player_observations'])):
            self.episode_file.write(self.obs_vec_compress(obs['player_observations'][i]['vectorized']) + "\n")
            # self.episode_file.write(self.pyhanabi_compress(str(obs['player_observations'][i]['pyhanabi'])) + "\n")


    def write_action(self, action):
        self.episode_file.write(str(action) + "\n")


    def reset(self, seed=None, options=None):
        self.seed(seed=seed)
        obs = super().reset()
        if self.record_episodes:
            self.write_obs(obs)
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])
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
        if self.record_episodes:
            self.write_action(move)
            self.write_obs(obs)
            if done:
                self.episode_file.write("-1\n")
        obs = np.array(obs['player_observations'][obs['current_player']]['vectorized'])

        truncate = False
        return obs, reward, done, truncate, info
