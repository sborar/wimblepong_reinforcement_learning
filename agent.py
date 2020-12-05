#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Sheetal Borar and Hossein Firooz
# Created Date: 18 Nov 2020
# =============================================================================
"""The file contains the agent used for playing Pong with Pixels"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import torch
import torch.nn as nn

from agent47_policy import Policy
from agent47_utils import discount_rewards


class Agent(object):
    def __init__(self):
        self.name = 'Agent47'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        self.previous_frame = None
        self.gamma = 0.9
        self.value_loss_coef = .5
        self.entropy_coef = 1e-2
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=7e-4, eps=1e-5, alpha=0.99)
        self.state_values = []
        self.rewards = []
        self.action_probs = []
        self.entropies = []

    def reset(self):
        self.previous_frame = None

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def load_model(self, path='model.mdl'):
        try:
            weights = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(weights, strict=False)
            print('Loaded model successfully:', path)
        except:
            print('Load model failed:', path)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def preprocessing(self, state):
        # taking mean across colour channel to make the image grayscale
        # reducing image size
        state = state[::2, ::2].mean(axis=-1)
        state = np.expand_dims(state, axis=-1)
        if self.previous_frame is None:
            self.previous_frame = state
        # stacking previous frame and the current frame
        stacked_states = np.concatenate((self.previous_frame, state), axis=-1)
        stacked_states = torch.from_numpy(stacked_states).float().unsqueeze(0)
        stacked_states = stacked_states.transpose(1, 3)
        self.previous_frame = state
        return stacked_states

    def get_action_train(self, observation):
        x = self.preprocessing(observation).to(self.device)
        dist, entropy, state_value = self.policy(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, entropy, state_value

    def get_action(self, observation):
        x = self.preprocessing(observation).to(self.device)
        dist, _, _ = self.policy(x)
        return torch.argmax(dist.probs).item()

    def store_outcome(self, state_value, reward, action_prob, entropy):
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.action_probs.append(action_prob)
        self.entropies.append(entropy)

    def episode_finished(self, episode_number):
        state_values = torch.stack(self.state_values, dim=0).squeeze().to(self.device)
        returns = discount_rewards(torch.tensor(self.rewards, device=self.device, dtype=torch.float), self.gamma)
        action_probs = torch.stack(self.action_probs, dim=0).squeeze().to(self.device)
        entropies = torch.stack(self.entropies, dim=0).squeeze().to(self.device)

        # empty the arrays
        self.state_values, self.rewards, self.action_probs, self.entropies = [], [], [], []

        # calculate advantage
        advantages = returns.detach() - state_values

        # calculate loss for policy, entropy and value
        policy_loss = -(advantages.detach() * action_probs).mean()
        entropy_loss = -(self.entropy_coef * entropies).mean()
        value_loss = (self.value_loss_coef * advantages.pow(2)).mean()
        loss = policy_loss + entropy_loss + value_loss

        # back prop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), .5)
        self.optimizer.step()

        # reduce entropy coef
        self.entropy_coef = 0.1 ** ((episode_number // 5000) + 1)
