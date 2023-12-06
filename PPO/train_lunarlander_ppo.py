import os
import glob
import time
from datetime import datetime
import warnings
import torch
import numpy as np
import gym
from ppo_agent import PPO

warnings.filterwarnings("ignore")

def train():
    has_continuous_action_space = False
    max_ep_len = 300  # max timesteps in one episode
    max_training_timesteps = int(1e6)  # break training loop if timeteps > max_training_timesteps
    checkpt = int(5e4)  # save model frequency (in num timesteps)
    update_timestep = max_ep_len * 3  # update policy every n timesteps
    k_epochs = 30  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network
    rewards = []  # list containing scores from each episode
    env = gym.make("LunarLander-v2")

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n
    print("State: ", state_dim,"\t\tAction: ", action_dim)
    # Setup logging directory
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create the directory if it doesn't exist
    log_f_name = log_dir + "/training_log.csv"  # Log file name

    # Setup model saving directory
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    checkpoint_path = directory + "/LunarLander_ppo_weights.pth"  # Checkpoint file path

    # Initialize the PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip)

    # Record the start time
    start_time = datetime.now().replace(microsecond=0)

    # Open the log file for writing
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward, average reward, 100-episode running mean reward\n')

    # Initialize time_step and episode count
    time_step = 0
    episode_num = 1

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _= env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # Save the model at checkpoints
            if time_step % checkpt == 0:
                print("Saving the model............................................")
                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break

        # Store and log episode reward
        rewards.append(current_ep_reward)
        avg_reward = np.mean(rewards)  # Average reward
        avg_reward = round(avg_reward, 4)
        mean_reward_100 = np.mean(rewards[-100:])  # Running mean of last 100 episodes
        mean_reward_100 = round(mean_reward_100, 4)
        current_ep_reward = round(current_ep_reward, 4)
        log_f.write('{},{},{},{}\n'.format(episode_num, current_ep_reward, avg_reward, mean_reward_100))
        log_f.flush()  # Flush the log file buffer
        # Print episode statistics
        print(
            "Episode : {} \t\t Reward: {} \t\t Average Reward : {} \t\t 100-episode running mean: {} \t\tTimestep: {} ".format(
                episode_num,
                current_ep_reward,
                avg_reward,
                mean_reward_100, time_step))
        if mean_reward_100 >= 200:
            print("Solved in {} episodes".format(episode_num))
            break
        episode_num += 1  # Increment episode counter

    log_f.close()
    env.close()

    print("Saving the model............................................")
    ppo_agent.save(checkpoint_path)

    # Calculate and print total training time
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time (minutes): ", (end_time - start_time).total_seconds() / 60.0)


if __name__ == '__main__':
    train()







