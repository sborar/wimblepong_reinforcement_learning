import wimblepong
import gym
from bn_agent import Agent
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 2000000

player = Agent(env, player_id=1)
env.set_names(player.get_name())
player.load_model(path = 'train_weights_bn_coef/0.890_finetune.pth')

print("Cuda:", torch.cuda.is_available())
print("Training")

run_title = "title"
wins = 0
frames_seen = 0
scores = [0 for _ in range(1000)]
game_lengths = [0 for _ in range(1000)]
running_state_values = [0 for _ in range(10000)]
running_entropies = [0 for _ in range(10000)]
running_action_probs = [0 for _ in range(10000)]

highest_running_winrate = 0
save_every = 10000

perf_plotting = []
game_length_plotting = []
state_value_plotting = []
entropy_plotting = []
action_prob_plotting = []

for episode_number in range(episodes):
    timesteps = 0
    done = False
    observation = env.reset()
    action_dist = [0,0,0]
    while not done:
        frames_seen += 1

        action, action_prob, entropy, state_value = player.get_action_train(observation)

        running_state_values.pop(0)
        running_state_values.append(state_value.item())

        running_entropies.pop(0)
        running_entropies.append(entropy.item())

        running_action_probs.pop(0)
        running_action_probs.append(action_prob.item())

        action_dist[action] += 1

        observation, reward, done, _ = env.step(action.detach())

        if reward == 10:
            wins += 1

        player.store_outcome(state_value, reward, action_prob, entropy)
            
        # Store total episode reward
        timesteps += 1

        if not args.headless:
            env.render()

    if reward == 10:
        scores.append(1)
        scores.pop(0)
    else:
        scores.append(0)
        scores.pop(0)

    game_lengths.pop(0)
    game_lengths.append(timesteps)
    
    run_avg = np.mean(np.array(scores))
    game_length_avg = np.mean(np.array(game_lengths))
    state_value_avg = np.mean(np.array(running_state_values))
    entropy_avg = np.mean(np.array(running_entropies))
    action_prob_avg = np.mean(np.array(running_action_probs))

    if  (run_avg - 0.01)  > highest_running_winrate:
        highest_running_winrate = run_avg
        torch.save(player.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate"+run_title+".pth")

    player.episode_finished(episode_number)
    player.reset()


    perf_plotting.append((episode_number,run_avg))
    game_length_plotting.append((episode_number,game_length_avg))
    state_value_plotting.append((episode_number,state_value_avg))
    entropy_plotting.append((episode_number,entropy_avg))
    action_prob_plotting.append((episode_number, action_prob_avg))

    if episode_number  % save_every == 0:
        torch.save(player.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate_at_"+str(episode_number)+"_episodes_"+run_title+".pth")

        df1 = pd.DataFrame(perf_plotting)
        df1.to_csv("./plot_data/"+run_title+str(episode_number)+"_perf_plotting.csv")

        df2 = pd.DataFrame(game_length_plotting)
        df2.to_csv("./plot_data/"+run_title+str(episode_number)+"_game_length_plotting.csv")

        df3 = pd.DataFrame(state_value_plotting)
        df3.to_csv("./plot_data/"+run_title+str(episode_number)+"_state_value_plotting.csv")

        df4 = pd.DataFrame(entropy_plotting)
        df4.to_csv("./plot_data/"+run_title+str(episode_number)+"_entropy_plotting.csv")

        df5 = pd.DataFrame(action_prob_plotting)
        df5.to_csv("./plot_data/"+run_title+str(episode_number)+"_action_prob_plotting.csv")

    print("("+run_title+") Episode over:",str(episode_number),"WR:",str(wins/(episode_number+1)),"wins:", wins, "steps", str(timesteps) ,"frames seen:", frames_seen, "action dist", action_dist, "highest winrate", highest_running_winrate, "current winrate:", run_avg)
torch.save(player.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate_at_end_"+run_title+"_.pth")