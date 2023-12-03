import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_rewards():
    df = pd.read_csv('training_logs/training_log.csv', header=0)
    if '100 episode running mean reward' in df.columns:
        running_mean_rewards = df['100 episode running mean reward'].to_numpy()
    else:
        print("Column not found in DataFrame.")
    rewards = df['reward'].to_numpy()
    running_mean_rewards = df['100 episode running mean reward'].to_numpy()
    episodes = df['episode'].to_numpy()
    plt.plot(episodes, rewards, label='reward')
    plt.plot(episodes, running_mean_rewards, label='100-episode running mean reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('training_log.png')
    plt.close()


def plot_testing_rewards():

    df = pd.read_csv('testing_logs/testing_log.csv', header=0)
    rewards = df['reward'].to_numpy()
    avg_rewards = df['average reward'].to_numpy()
    episodes = df['episode'].to_numpy()
    plt.plot(episodes, rewards, label='reward')
    plt.plot(episodes, avg_rewards, label='average reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('testing_log.png')
    plt.close()

if __name__ == '__main__':
    plot_training_rewards()
    plot_testing_rewards()

