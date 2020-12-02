import argparse
import base64
import glob
import gym
import io
from IPython import display as ipythondisplay
from IPython.display import HTML
from gym import logger as gymlogger
from gym.wrappers import Monitor
import wimblepong
from bn_agent import Agent

gymlogger.set_level(40)  # error only

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
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
player.load_model(path='train_weights_bn_coef/0.890_finetune.pth')

# Housekeeping
states = []
win1 = 0

for i in range(0,episodes):
    done = False
    observation = env.reset()
    while not done:
        action = player.get_action(observation)
        observation, reward, done, info = env.step(action)

        if reward == 10:
            wins += 1
        if i > 50:
            env.render()

    player.reset()
    print("Episode over:", i, "wins:", wins, "winrate:", wins / (i + 1))
    if i % 5 == 4:
        env.switch_sides()


