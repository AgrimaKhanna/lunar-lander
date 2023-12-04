import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent import Agent

def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    env = gym.make('LunarLander-v2')
    agent = Agent(state_size=8, action_size=4, seed=0)
    scores = []  # list containing scores from each episode
    running_mean_100 = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    # Setup logging directory
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create the directory if it doesn't exist
    log_f_name = log_dir + "/training_log.csv"  # Log file name

    # Setup model saving directory
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    checkpoint_path = directory + "/LunarLander_dqn_weights.pth"  # Checkpoint file path

    # Open the log file for writing
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward, average reward, 100-episode running mean reward\n')

    for episode_num in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)  # save most recent score
        running_mean_100 = np.mean(scores[-100:]) # calculate running mean of last 100 scores
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('Episode {}\t\t Score: {:.4f} \t\t Average Score: {:.4f} \t\t 100-episode running mean: {:.4f}'.format(episode_num, score, np.mean(scores), running_mean_100))
        log_f.write('{},{},{},{}\n'.format(episode_num, score, np.mean(scores), running_mean_100))
        log_f.flush()  # Flush the log file buffer
        if np.mean(running_mean_100) >= 200.0:
            print('\nEnvironment solved in {:d} episodes! \t 100-episode running mean: {:.2f}'.format(episode_num - 100, running_mean_100))
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
            print("Saving the model............................................")
            break
if __name__ == "__main__":
    train()





