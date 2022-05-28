"""models for a2c agents"""
import torch
from torch import nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, l_obs, n_act, hidden_size=32):
        super(ActorCritic, self).__init__()
        self.l_obs = l_obs
        self.n_act = n_act
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(self.l_obs, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Linear(self.hidden_size, self.n_act)
        self.critic = nn.Linear(self.hidden_size, 1)

        self.fc.apply(weight_init)
        self.actor.apply(weight_init)
        self.critic.apply(weight_init)

    def forward(self, inputs):
        x = self.fc(inputs)
        pi = self.actor(x)
        vi = self.critic(x)

        return pi, vi


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)
