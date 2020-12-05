#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Sheetal Borar and Hossein Firooz
# Created Date: 18 Nov 2020
# =============================================================================
"""The file contains the policy used by the agent"""
# =============================================================================
# Imports
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        h, w = 100, 100

        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=32,
                               kernel_size=8,
                               stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def output_size(w, h):
            convw = conv2d_size_out(w, 8, 4)
            convw = conv2d_size_out(convw, 4, 2)
            convw = conv2d_size_out(convw, 3, 1)

            convh = conv2d_size_out(h, 8, 4)
            convh = conv2d_size_out(convh, 4, 2)
            convh = conv2d_size_out(convh, 3, 1)

            return convw * convh * 32

        linear_input_size = output_size(w, h)

        self.linear1 = nn.Linear(linear_input_size, 512)

        self.fc1_actor = torch.nn.Linear(512, 256)
        self.fc1_critic = torch.nn.Linear(512, 256)

        self.fc2_probs = torch.nn.Linear(256, 3)
        self.fc2_value = torch.nn.Linear(256, 1)

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
