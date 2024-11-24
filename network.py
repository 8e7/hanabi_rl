import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces

class ObservationEmbedding(BaseFeaturesExtractor):
    """Turn the input vector into a low dimensional embedding.
    :param observation_space: (gym.Space)
    :param embedding_dim: (int) Dimension of the embedding (k).
    """

    def __init__(self, observation_space: spaces.MultiBinary, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self.input_dim = observation_space.shape[0]
        self.hidden_dim = features_dim

        self.network = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.features_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        #batch_size = observation.shape[0] 
        return self.network(observation)

