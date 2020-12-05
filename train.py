#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Sheetal Borar and Hossein Firooz
# Created Date: 18 Nov 2020
# =============================================================================
"""The file contains the training code used to train the agent"""
# =============================================================================
# Imports
# =============================================================================

import argparse
import gym
import numpy as np
import pandas as pd
import torch
import wimblepong
from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--load_model_path", type=str, help="Path to load an existing model")
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# check if gpu is available
print("Cuda:", torch.cuda.is_available())
print("Start Training")

# saving folders
model_path = "./train_weights/"
plot_data_path = "./plot_data/"

# Define Agent
player = Agent()
env.set_names(player.get_name())

# load existing model
if args.load_model_path:
    player.load_model(path=args.load_model_path)

# initialize variables
episodes = 2000000
wins = 0
frames_seen = 0
scores = [0 for _ in range(100)]
highest_running_winrate = 0
save_every = 10000

win_rates = []

for episode_number in range(episodes):
    timesteps = 0
    done = False
    observation = env.reset()
    while not done:

        action, action_prob, entropy, state_value = player.get_action_train(observation)
        observation, reward, done, _ = env.step(action.detach())

        if reward == 10:
            wins += 1

        player.store_outcome(state_value, reward, action_prob, entropy)

        # Housekeeping
        timesteps += 1
        frames_seen += 1

        if not args.headless:
            env.render()

    if reward == 10:
        scores.append(1)
        scores.pop(0)
    else:
        scores.append(0)
        scores.pop(0)

    # update policy
    player.episode_finished(episode_number)
    player.reset()

    # print and save things
    avg_win_rate_100_eps = np.mean(np.array(scores))

    if (avg_win_rate_100_eps - 0.01) > highest_running_winrate:
        highest_running_winrate = avg_win_rate_100_eps
        player.save_model(model_path + str(highest_running_winrate) + "_winrate.mdl")

    win_rates.append((episode_number, frames_seen, avg_win_rate_100_eps))

    if episode_number % save_every == 0:
        df1 = pd.DataFrame(win_rates)
        df1.to_csv(plot_data_path + str(episode_number) + "_win_rates.csv")

    print(
        "Episode:", str(episode_number), "Overall WR:", str(wins / (episode_number + 1)), "Wins:", wins,
        "steps:", str(timesteps), "frames seen:", frames_seen, "highest winrate:",
        highest_running_winrate, "100 episode average winrate:", avg_win_rate_100_eps)

player.save_model(model_path + str(highest_running_winrate) + "_winrate_at_end.mdl")
