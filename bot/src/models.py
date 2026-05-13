import torch
import torch.nn as nn
import torch.nn.functional as F
import bot.config as config
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CryptoCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CryptoCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.conv1 = nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc_input_dim = self._get_conv_output_shape()
        self.fc1 = nn.Linear(self.fc_input_dim, features_dim)
        
        if config.DEBUG: 
            print(f"[TRACE] CNN Acoplada com Sucesso. Input: {config.IMG_SIZE} | Flatten: {self.fc_input_dim}")

    def _get_conv_output_shape(self):
        dummy_input = torch.zeros(1, 1, config.IMG_SIZE[0], config.IMG_SIZE[1])
        x = self.pool(F.relu(self.conv1(dummy_input)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))            # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))            # 16x16 -> 8x8
        return x.numel()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(observations)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))
