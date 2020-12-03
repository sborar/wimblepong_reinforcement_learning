import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 9 * 9, 512)

        self.fc2_actor = torch.nn.Linear(512, 256)
        self.fc2_critic = torch.nn.Linear(512, 256)

        self.fc3_probs = torch.nn.Linear(256, 3)
        self.fc3_value = torch.nn.Linear(256, 1)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, inputs):
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))

        # actor layer
        x_actor = self.fc2_actor(x)
        x_actor = F.relu(x_actor)
        x_mean = self.fc3_probs(x_actor)
        x_probs = F.softmax(x_mean, dim=-1)
        actor_dist = Categorical(x_probs)
        entropy = actor_dist.entropy()

        # critic layer
        x_critic = self.fc2_critic(x)
        x_critic = F.relu(x_critic)
        state_value = self.fc3_value(x_critic)

        return actor_dist, entropy, state_value


class Agent(object):
    def __init__(self):
        self.name = 'Batch-norm'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        self.previous_frame = None
        self.gamma = 0.9
        self.value_loss_coef = .5
        self.entropy_coef = 1e-2
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=0.0001, eps=1e-5, alpha=0.99)
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

    def load_model(self, path):
        try:
            weights = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(weights, strict=False)
            print('Loaded model successfully:', path)
        except:
            print('Load model failed:', path)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def preprocessing(self, state):
        state = state[::2, ::2].mean(axis=-1)
        state = np.expand_dims(state, axis=-1)
        if self.previous_frame is None:
            self.previous_frame = state
        stacked_states = np.concatenate((self.previous_frame, state), axis=-1)
        stacked_states = torch.from_numpy(stacked_states).float().unsqueeze(0)
        stacked_states = stacked_states.transpose(1, 3)
        self.previous_frame = state
        return stacked_states/255.0

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
        self.entropy_coef = 0.1**((episode_number//5000)+1)
