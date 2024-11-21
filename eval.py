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

from utils import bcolors

register(
    id='hanabi-eval',
    entry_point='envs:HanabiEvalEnv',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'model_path': 'models/PPO'
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

def visualize(model_list):
    """Play a game using model. 2 players."""

    def print_state(state):
        """Print some basic information about the state."""
        print()
        print(bcolors.HEADER + "New Step" + bcolors.ENDC)
        print("Current player: {}".format(state.cur_player()))
        print(state)
        print("")
        # Example of more queries to provide more about this state
        """
        print("### Information tokens: {}".format(state.information_tokens()))
        print("### Life tokens: {}".format(state.life_tokens()))
        print("### Fireworks: {}".format(state.fireworks()))
        print("### Deck size: {}".format(state.deck_size()))
        print("### Discard pile: {}".format(str(state.discard_pile())))
        print("### Player hands: {}".format(str(state.player_hands())))
        """

    def print_observation(observation):
        """Print some basic information about an agent observation."""
        print(bcolors.WARNING + "--- Observation ---" + bcolors.ENDC)
        print(observation)

        print("### Information about the observation retrieved separately ###")
        print("### Current player, relative to self: {}".format(
                observation.cur_player_offset()))
        print("### Observed hands: {}".format(observation.observed_hands()))
        print("### Card knowledge: {}".format(observation.card_knowledge()))
        move_string = "### Last moves:"
        for move_tuple in observation.last_moves():
            move_string += " {}".format(move_tuple)
        print(move_string)
        print(bcolors.WARNING + "--- EndObservation ---" + bcolors.ENDC)
        """
        print("### Discard pile: {}".format(observation.discard_pile()))
        print("### Fireworks: {}".format(observation.fireworks()))
        print("### Deck size: {}".format(observation.deck_size()))
        
        print("### Information tokens: {}".format(observation.information_tokens()))
        print("### Life tokens: {}".format(observation.life_tokens()))
        print("### Legal moves: {}".format(observation.legal_moves()))
        """

    def print_encoded_observations(encoder, state, num_players):
        print("--- EncodedObservations ---")
        print("Observation encoding shape: {}".format(encoder.shape()))
        print("Current actual player: {}".format(state.cur_player()))
        for i in range(num_players):
            print("Encoded observation for player {}: {}".format(
                    i, encoder.encode(state.observation(i))))
        print("--- EndEncodedObservations ---")

    game = pyhanabi.HanabiGame({"players": 2, "random_start_player": True})
    print(game.parameter_string(), end="")
    obs_encoder = pyhanabi.ObservationEncoder(
            game, enc_type=pyhanabi.ObservationEncoderType.CANONICAL)

    state = game.new_initial_state()
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        print_state(state)

        observation = state.observation(state.cur_player())
        #print_observation(observation)
        #print_encoded_observations(obs_encoder, state, game.num_players())
        obs = obs_encoder.encode(observation)

        legal_moves = state.legal_moves()
        action, _state = model_list[state.cur_player()].predict(obs, deterministic=True)
        action = action % len(legal_moves)
        print("Chose move: {}".format(legal_moves[action]))
        state.apply_move(legal_moves[action])

    print("")
    print(bcolors.BOLD + "Game done. Terminal state:" + bcolors.ENDC)
    print("")
    print(state)
    print("")
    print("score: {}".format(state.score()))
   
def evaluate(model_list, env, eval_num=100, vis=False):
    
    if len(model_list) != env_config["players"]:
        print(f"Length of model_list is not equal to {env_config['players']}")
        model_list = model_list[:env_config["players"]]
        model_list.extend([model_list[-1]] * (env_config["players"] - len(model_list)))

    tot_score = 0
    for iteration in range(eval_num):
        done = False
        obs, info = env.reset(seed=random.randint(0, 100000000))
        while not done:
            player = env.state.cur_player()
            action, _state = model_list[player].predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        score = env.state.score()
        tot_score += score
    print(f"Average score: {tot_score / eval_num}")

    if vis:
        visualize(model_list)
    return tot_score / eval_num


if __name__ == "__main__":
    env = gym.make('hanabi-eval', config=env_config)

    model = model_config['algorithm'].load(model_config['model_path'])
    evaluate(model, env, eval_num=100, vis=True)
