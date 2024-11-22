import numpy as np
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class constant:
    obs_dim_2p = 658

def parse_obs(obs_str):
    # turn hex string to binary 0-1 array
    obs_str = bin(int(obs_str, 16))[2:].zfill(constant.obs_dim_2p)
    return np.array([int(x) for x in obs_str])

def read_episode_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    sa = []
    episode_count = 0
    # every 4 lines is a state-action pair
    # line 1: current player
    # line 2: observation of player 0 (hex of binary vec of length = obs_dim_2p)
    # line 3: observation of player 1 (same)
    # line 4: action (-1 if terminal)
    for i in range(0, len(lines), 4):
        current_player = int(lines[i])
        obs_0 = parse_obs(lines[i+1])
        obs_1 = parse_obs(lines[i+2])
        action = int(lines[i+3])
        sa.append((current_player, obs_0, obs_1, action))
        if action == -1:
            episode_count += 1
    return sa, episode_count
    

        