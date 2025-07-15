import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        return x + self.up_proj(self.activation(self.down_proj(x)))
