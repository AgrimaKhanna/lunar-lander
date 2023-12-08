import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gym
from ppo_continuous_agent import PPO  # Importing the PPO algorithm


def test():

    # Setting various parameters for the test
    max_ep_len = 1500  # Maximum length of each episode
    action_std = 0.1  # Standard deviation of action distribution
    render = True  # Whether to render the environment
    frame_delay = 0  # Delay between frames when rendering
    total_test_episodes = 200  # Total number of testing episodes
    k_epochs = 30  # Number of epochs for updating policy
    eps_clip = 0.2  # Clipping parameter for PPO
    gamma = 0.99  # Discount factor for reward
    lr_actor = 0.0003  # Learning rate for the actor network
    lr_critic = 0.001  # Learning rate for the critic network

    # Create the environment
    env = gym.make("LunarLanderContinuous-v2", render_mode = 'human')  # Initialize the Gym environment
    state_dim = env.observation_space.shape[0]  # State dimensionality
    action_dim = env.action_space.shape[0]  # Action dimensionality

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, action_std)

    # Load the pre-trained model
    checkpoint_path = "models/LunarLander_ppo_continuous_weights.pth"
    ppo_agent.load(checkpoint_path)

    # Setup logging
    log_dir = "testing_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create log directory if it doesn't exist
    log_f_name = log_dir + "/testing_log.csv"
    log_f = open(log_f_name, "w+")  # Open log file for writing
    log_f.write('episode,reward, average reward\n')  # Write log file headers

    # Initialize rewards list
    rewards = []

    # Loop over test episodes
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0  # Initialize reward for the episode
        state = env.reset()  # Reset the environment

        # Iterate over each step in the episode
        for t in range(1, max_ep_len+1):
            try:
                action = ppo_agent.select_action(state)  # Select action based on current state
            except:
                state = state[0]  # Handle any exceptions (e.g., invalid state shape)
                action = ppo_agent.select_action(state)  # Retry selecting action
            state, reward, done, _, _ = env.step(action)  # Take the action in the environment
            ep_reward += reward  # Accumulate reward

            if render:
                env.render()  # Render the environment
                time.sleep(frame_delay)  # Add delay between frames

            if done:
                break  # Exit loop if episode is done

        ppo_agent.buffer.clear()  # Clear PPO buffer
        rewards.append(ep_reward)  # Append episode reward to rewards list

        # Calculate and log average reward
        avg_reward = np.mean(rewards)
        avg_reward = round(avg_reward, 4)
        ep_reward = round(ep_reward, 4)
        log_f.write('{},{},{}\n'.format(ep, ep_reward, avg_reward))  # Write to log file
        log_f.flush()  # Flush the log file buffer
        # Print episode stats
        print('Episode: {} \t\t Reward: {}\t\t Average Reward: {}'.format(ep, ep_reward, avg_reward))

    env.close()  # Close the environment
    print("Average test reward : " + str(np.mean(rewards)))  # Print average reward


if __name__ == '__main__':

    test()  # Run the test
