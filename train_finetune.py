"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image
from agent import Agent as FishAgent
import pandas as pd
from test_agents.KarpathyNotTrained.agent import Agent as KarpathyAgent
from test_agents.SomeAgent.agent import Agent as SomeAgent
from test_agents.SomeOtherAgent.agent import Agent as SomeOtherAgent
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 2000000

# Define the player IDs for both SimpleAI agents
learner_id = 1
teacher_id = 3 - learner_id

learner_validator_id = 1
validator_id = 3 - learner_validator_id

learner = FishAgent()
teacher = FishAgent()

filename = 'model.mdl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.load(filename, map_location=device)
learner.policy.load_state_dict(weights, strict=False) #Load weights for learner
teacher.policy.load_state_dict(weights, strict=False) #Load weights for teacher
teacher.policy.eval()
teacher.test = True

learner_validator = FishAgent()
validator_1 = wimblepong.SimpleAi(env, validator_id)
validator_2 = SomeAgent()
validator_2.load_model()
validator_3 = SomeOtherAgent()
validator_3.load_model()
validator_4 = KarpathyAgent()
validator_4.load_model()

validators = [validator_1, validator_2, validator_3, validator_4]
validator = validator_1


env.set_names(learner.get_name(), teacher.get_name())

def validation_run(env,n_games=100):
    learner_validator.policy.load_state_dict(learner.policy.state_dict())
    learner_validator.policy.eval()
    env.set_names(learner_validator.get_name(), validator_1.get_name())

    win1 = 0

    for j in range(0, n_games):
        done = False
        ob1, ob2 = env.reset()
        action_dist = [0,0,0]
        while not done:
            action1,_,_,_ = learner_validator.get_action_train(ob1)
            action2 = validator_1.get_action(ob2)
            action_dist[action1] += 1

            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

            if not args.headless:
                env.render()

            if rew1 == 10:
                win1 += 1
        learner_validator.reset()

        print("Validation episode over:",j,"WR:",str(win1/(j+1)),"wins:", win1, "action dist", action_dist)
    
    return win1/n_games

run_title = "title"
highest_running_winrate = 0
highest_validation_winrate = 0
save_every = 1000
validate_every = 1000
validation_winrate = 0
game_lengths = [0 for _ in range(1000)]
scores = [0 for _ in range(1000)]
running_state_values = [0 for _ in range(10000)]
running_entropies = [0 for _ in range(10000)]
running_action_probs = [0 for _ in range(10000)]
validation_scores = [0 for _ in range(10)]
win1 = 0
frames_seen = 0
opponent = "teacher"
change_opp_every = 10

perf_plotting = []
game_length_plotting = []
state_value_plotting = []
entropy_plotting = []
action_prob_plotting = []
validation_score_plotting = []

for episode_number in range(0,episodes):
    timesteps = 0
    done = False
    ob1, ob2 = env.reset()
    action_dist = [0,0,0]
    while not done:
        frames_seen += 1
        # Get the actions from both SimpleAIs
        action1, action_prob, entropy, state_value = learner.get_action_train(ob1)
        running_state_values.pop(0)
        running_state_values.append(state_value.item())

        running_entropies.pop(0)
        running_entropies.append(entropy.item())

        running_action_probs.pop(0)
        running_action_probs.append(action_prob.item())
        action_dist[action1] += 1
        if opponent == "teacher":
            action2,_,_,_ = teacher.get_action_train(ob2)
        else:
            action2 = validator.get_action(ob2)
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        if rew1 == 10:
            win1 += 1

        learner.store_outcome(state_value, rew1, action_prob, entropy)
            
        # Store total episode reward
        timesteps += 1

        if not args.headless:
            env.render()

    if rew1 == 10:
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
        torch.save(learner.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate"+str(highest_validation_winrate)+"_valwinrate_"+run_title+".pth")
        teacher.policy.load_state_dict(learner.policy.state_dict())
        teacher.policy.eval()
        print("Teacher upgraded")

    learner.episode_finished()
    learner.reset()
    teacher.reset()

    perf_plotting.append((episode_number,run_avg))
    game_length_plotting.append((episode_number,game_length_avg))
    state_value_plotting.append((episode_number,state_value_avg))
    entropy_plotting.append((episode_number,entropy_avg))
    action_prob_plotting.append((episode_number, action_prob_avg))

    if episode_number % change_opp_every == 0:
        if opponent == "teacher":
            opponent = "validator"
            validator = np.random.choice(validators)
            env.set_names(learner.get_name(), validator.get_name())
            print("Swithced to:", validator.get_name())
        else:
            print("Swithced to teacher")
            opponent = "teacher"
            env.set_names(learner.get_name(), teacher.get_name())
            

    if episode_number % validate_every == 0:
        validation_winrate = validation_run(env)
        validation_scores.pop(0)
        validation_scores.append(validation_winrate)

        validation_avg = np.mean(np.array(validation_scores))
        validation_score_plotting.append((episode_number,validation_avg))
        
        
        if validation_winrate > highest_validation_winrate:
            highest_validation_winrate = validation_winrate
            torch.save(learner.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate"+str(highest_validation_winrate)+"_valwinrate_"+run_title+".pth")
        env.set_names(learner.get_name(), teacher.get_name())
        print("Validation done")

    if episode_number  % save_every == 0:
        torch.save(learner.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate_"+str(highest_validation_winrate)+"_valwinrate_"+str(episode_number)+"_episodes_"+run_title+".pth")
        df1 = pd.DataFrame(perf_plotting)
        df1.to_csv("./plot_data"+run_title+str(episode_number)+"_perf_plotting.csv")

        df2 = pd.DataFrame(game_length_plotting)
        df2.to_csv("./plot_data"+run_title+str(episode_number)+"_game_length_plotting.csv")

        df3 = pd.DataFrame(state_value_plotting)
        df3.to_csv("./plot_data"+run_title+str(episode_number)+"_state_value_plotting.csv")

        df4 = pd.DataFrame(entropy_plotting)
        df4.to_csv("./plot_data"+run_title+str(episode_number)+"_entropy_plotting.csv")

        df5 = pd.DataFrame(action_prob_plotting)
        df5.to_csv("./plot_data"+run_title+str(episode_number)+"_action_prob_plotting.csv")

        df6 = pd.DataFrame(validation_score_plotting)
        df6.to_csv("./plot_data"+run_title+str(episode_number)+"_validation_score_plotting.csv")

    print("("+run_title+") Episode over:",str(episode_number),"WR:",str(win1/(episode_number+1)),"wins:", win1, "steps", str(timesteps) ,"frames seen:", frames_seen, "action dist", action_dist, "highest winrate", highest_running_winrate, "current winrate:", run_avg, "highest val winrate", highest_validation_winrate, "last val winrate", validation_winrate)

torch.save(learner.policy.state_dict(), "./train_weights/"+str(highest_running_winrate)+"_winrate_at_end_"+run_title+"_.pth")