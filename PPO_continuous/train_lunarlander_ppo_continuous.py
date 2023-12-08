import gym
import torch
import numpy as np
import os
import glob
import time
import warnings
from datetime import datetime
from ppo_continuous_agent import PPO  # Import PPO class

# Suppress warnings
warnings.filterwarnings("ignore")


# Define the train function
def train():
    # Setting various parameters for training
    max_episode_length = 300  # Maximum length of an episode
    max_training_timesteps = int(3e6)  # Total training timesteps
    update_timestep = max_episode_length * 3  # Timesteps per policy update
    k_epochs = 80  # Number of epochs per policy update
    eps_clip = 0.2  # Clipping parameter for PPO
    gamma = 0.99  # Discount factor
    lr_actor = 0.0003  # Learning rate for the actor network
    lr_critic = 0.001  # Learning rate for the critic network
    action_std = 0.6  # Initial standard deviation of action distribution
    action_std_decay_rate = 0.05  # Decay rate of action standard deviation
    min_action_std = 0.1  # Minimum standard deviation of action
    action_std_decay_freq = int(2.5e5)  # Frequency of decay steps
    rewards = []  # List to store episode rewards
    checkpt = int(5e4)  # Checkpoint frequency

    # Initialize the Gym environment
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]  # State dimensionality
    action_dim = env.action_space.shape[0]  # Action dimensionality

    # Setup logging directory
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create the directory if it doesn't exist
    log_f_name = log_dir + "/training_log.csv"  # Log file name

    # Setup model saving directory
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist
    checkpoint_path = directory + "/LunarLander_ppo_continuous_weights.pth"  # Checkpoint file path

    # Initialize the PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std)

    # Record the start time
    start_time = datetime.now().replace(microsecond=0)

    # Open the log file for writing
    log_f = open(log_f_name, "w+")
    log_f.write('episode,reward, average reward, 100-episode running mean reward\n')

    # Initialize time_step and episode count
    time_step = 0
    episode_num = 1

    # Training loop
    while time_step <= max_training_timesteps:
        state = env.reset()  # Reset the environment to start a new episode
        current_ep_reward = 0  # Initialize reward for the current episode

        # Loop for each step in the episode
        for t in range(1, max_episode_length + 1):
            try:
                action = ppo_agent.select_action(state)  # Select an action based on current state
            except:
                state = state[0]  # Handle any exceptions
                action = ppo_agent.select_action(state)  # Retry selecting action
            state, reward, done, _, _ = env.step(action)  # Take the action in the environment
            ppo_agent.buffer.rewards.append(reward)  # Store reward
            ppo_agent.buffer.is_terminals.append(done)  # Store terminal state flag
            time_step += 1  # Increment timestep
            current_ep_reward += reward

            # Policy update
            if time_step % update_timestep == 0:
                ppo_agent.update()  # Update the policy

            # Decay action standard deviation
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Save the model at checkpoints
            if time_step % checkpt == 0:
                print("Saving the model............................................")
                ppo_agent.save(checkpoint_path)

            if done:
                break  # End the episode if done

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
            "Episode : {} \t\t Reward: {} \t\t Average Reward : {} \t\t 100-episode running mean: {}".format(episode_num,

                                                                                                            current_ep_reward,
                                                                                                             avg_reward,
                                                                                                             mean_reward_100))
        if mean_reward_100 >= 120:
            print("Solved in {} episodes".format(episode_num))
            break
        episode_num += 1  # Increment episode count

    # Close files and environment
    log_f.close()
    env.close()

    # Calculate and print total training time
    end_time = datetime.now().replace(microsecond=0)
    print("Total training time (minutes): ", (end_time - start_time).total_seconds() / 60.0)


if __name__ == '__main__':
    train()  # train the agent







