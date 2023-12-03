import matplotlib.pyplot as plt
import numpy as np
import os


def simple_plot(rewards, mean_rewards, epoch):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.plot(mean_rewards)
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.text(len(mean_rewards) - 1, mean_rewards[-1], str(mean_rewards[-1]))
    if epoch % 10 == 0:
        plt.show()

def plot_learning_curve(x, rewards, figure_file, algo, env_id):

    if not os.path.exists('./training_logs'):
        os.makedirs('./training_logs')

    running_avg = np.zeros(len(rewards))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(rewards[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(f'{algo} {env_id} Average 100 rewards')
    plt.savefig(figure_file)
    plt.close()