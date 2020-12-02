"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from bn_agent import Agent


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--filename", type=str, help="Weights to test", default=None)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 1000

# Define the player
player = Agent(env, player_id=1)

env.set_names("Policy gradient")
player.load_model(path='train_weights/0.965_winrate_at_100000_episodes_title.pth')

# Housekeeping
states = []
win1 = 0

observation = env.reset()

for i in range(0,episodes):
    done = False
    while not done:
        action = player.get_action(observation)
        observation, reward, done, info = env.step(action)
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if reward == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}. wins: {}".format(i, win1/(i+1), win1))
            if i % 5 == 4:
                env.switch_sides()

