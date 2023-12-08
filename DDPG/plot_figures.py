import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_testing_rewards(file_path):
    df = pd.read_csv('testing_logs/testing_log.csv', header=0)
    rewards = df['reward'].to_numpy()
    episodes = df['episode'].to_numpy()
    plt.plot(episodes, rewards, label='reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_testing_rewards("plots/testing_log.png")