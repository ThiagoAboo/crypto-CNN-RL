import torch
import torch.nn as nn
import torch.nn.functional as F
import bot.config as config

class CryptoCNN(nn.Module):
    def __init__(self, num_actions=3):
        super(CryptoCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc_input_dim = self._get_conv_output_shape()
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.out = nn.Linear(128, num_actions)
        
        if config.DEBUG_TRACE: 
            print(f"[TRACE] CNN Criada. Input: {config.IMG_SIZE} | Flatten: {self.fc_input_dim}")

    def _get_conv_output_shape(self):
        dummy_input = torch.zeros(1, 1, config.IMG_SIZE[0], config.IMG_SIZE[1])
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)
