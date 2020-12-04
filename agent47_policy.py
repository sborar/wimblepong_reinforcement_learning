import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32 * 9 * 9, 512)

        self.fc1_actor = torch.nn.Linear(512, 256)
        self.fc1_critic = torch.nn.Linear(512, 256)

        self.fc2_probs = torch.nn.Linear(256, 3)
        self.fc2_value = torch.nn.Linear(256, 1)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs / 255.0)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.linear1(x))

        # actor layer
        x_actor = F.relu(self.fc1_actor(x))
        x_mean = self.fc2_probs(x_actor)
        x_probs = F.softmax(x_mean, dim=-1)
        actor_dist = Categorical(x_probs)
        entropy = actor_dist.entropy()

        # critic layer
        x_critic = self.fc1_critic(x)
        x_critic = F.relu(x_critic)
        state_value = self.fc2_value(x_critic)

        return actor_dist, entropy, state_value
