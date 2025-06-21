import torch.nn as nn
import torch.nn.functional as f


# fully connected layer를 사용하기 위한 함수.
class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dimension = config.d_hidn
        self.network = config.d_ff

        self.layer1 = nn.Linear(self.dimension, self.network)
        self.layer2 = nn.Linear(self.network, self.dimension)

        self.active = f.relu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.active(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)

        return x
