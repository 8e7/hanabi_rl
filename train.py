import torch
from torch import cuda
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from typing import Callable

import wandb
from wandb.integration.sb3 import WandbCallback

from hanabi_learning_environment import pyhanabi

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

from network import ObservationEmbedding
from eval import evaluate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--retrain', default=False)

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"Using {device}")
register(
    id='hanabi-v0',
    entry_point='envs:HanabiEnvWrapper',
)
register(
    id='hanabi-eval',
    entry_point='envs:HanabiEvalEnv',
)

warnings.simplefilter(action='ignore', category=FutureWarning)

model_config = {
    'algorithm': PPO,
    'policy_network': 'MlpPolicy',
    'save_path': 'models/PPO_new',
    'run_id': 'PPO_new'
}
train_config = {
    'num_train_envs': 16,
    'training_steps': 65536,
    'n_steps': 256,
    'n_steps_testing': 640,
    'batch_size': 512, 
    'n_epochs': 2000,
    'learning_rate': 1e-4
}
env_config = {
    "colors":                   5,
        #"colors":                   2,
    "ranks":                    5,
    "players":                  2,
    "hand_size":                5,
    #"hand_size":                2,
    "max_information_tokens":   8,
    #"max_information_tokens":   3,
    "max_life_tokens":          3,
    #"max_life_tokens":          1,
    "observation_type":         pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value,
    "record_episodes": False,
    "episodes_file": "episodes"
}

def make_env():
    env = gym.make('hanabi-v0', config=env_config)
    return env

def train(model, eval_env, retry=False):
    if retry:
        best_avg = evaluate([model], eval_env, eval_num=100)
    else:
        best_avg = -1
    for epoch in range(train_config['n_epochs']):
        print("Epoch", epoch)
        model.learn(
            total_timesteps=train_config['training_steps'],
            reset_num_timesteps=False,
            #callback=WandbCallback(verbose=1),
            log_interval=16
        )

        avg_score = evaluate([model], eval_env, eval_num=30)
        approx_kl = model.logger.name_to_value['train/approx_kl']

        wandb.log(
            {"avg_score": avg_score,
            "epoch": epoch,
            "approx_kl": approx_kl}
        )
        if approx_kl > 0.02:
            print(f"Approx kl is {approx_kl}, Adjusting Learning rate")
            new_learning_rate = model.learning_rate * 0.95
            model.learning_rate = new_learning_rate
            model._setup_lr_schedule()
        elif approx_kl < 0.005:
            model.learning_rate = max(model.learning_rate, 1e-6)
            model._setup_lr_schedule()

        if avg_score > best_avg:
            best_avg = avg_score
            print("Saving Model")
            model.save(f"{model_config['save_path']}")

if __name__ == "__main__":
    args = parser.parse_args()
    run = wandb.init(
        project="RL_Final",
        config=train_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=model_config["run_id"],
        settings=wandb.Settings(_disable_stats=True)
    )

    train_env = SubprocVecEnv([make_env for _ in range(train_config['num_train_envs'])])
    eval_env = gym.make('hanabi-eval', config=env_config)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        activation_fn=torch.nn.Tanh
        #features_extractor_class=ObservationEmbedding,
        #features_extractor_kwargs=dict(features_dim=64)
    )
    if args.retrain:
        model = model_config['algorithm'].load(model_config['save_path'])
        print("Retraining")
    else:
        model = model_config['algorithm'](
            model_config['policy_network'],
            train_env,
            verbose=2,
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size'],
            learning_rate=train_config['learning_rate'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs/{model_config['run_id']}",
            device=device,
            ent_coef=0.01,
        )
   
    print(model.policy)
    train(model, eval_env, args.retrain)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100, deterministic=False)
    print(f"Mean reward: {mean_reward}, std_reward: {std_reward}")
