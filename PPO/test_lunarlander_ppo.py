import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gym
from ppo_agent import PPO

def test():

    max_ep_len = 300
    total_test_episodes = 10    # total num of testing episodes
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    env = gym.make("LunarLander-v2")

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    checkpoint_path = "models/lunarlander_ppo_weights.pth"
    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 4)))

    env.close()


if __name__ == '__main__':

    test()