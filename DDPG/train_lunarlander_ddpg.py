import csv
import time
from datetime import datetime
import gym
import numpy as np
import torch
import warnings
from ddpg_agent import Agent
import os

warnings.filterwarnings("ignore")
# Display the device being used
device = "cpu"

# Hyperparameters
num_episodes = 1000
load_checkpoint = False
success_threshold = 200.0
environment_name = 'LunarLanderContinuous-v2'
env = gym.make(environment_name)

agent_args = {
    'num_steps': 1000000,
    'buffer_size': 1000000,
    'min_buffer_size': 1000,
    'batch_size': 64,
    'start_steps': 5000,
    'hidden_layers': [400, 300],
    'discount_factor': 0.99,
    'policy_learning_rate': 0.0001,
    'critic_learning_rate': 0.001,
    'adam_epsilon': 1e-7
}

# Create the DDPG agent
agent = Agent(policy_lr=agent_args['policy_learning_rate'],
              critic_lr=agent_args['critic_learning_rate'],
              input_dims=env.observation_space.shape,
              tau=0.001, env_id=environment_name,
              batch_size=agent_args['batch_size'],
              hidden_dims=agent_args['hidden_layers'],
              n_actions=env.action_space.shape[0])

def train_agent():
    best_score = env.reward_range[0]
    score_history = []
    total_training_time = []
    running_mean_100 = []
    avg_scores = []
    eps_critic_losses, eps_actor_losses = [], []

    if load_checkpoint:
        agent.load_models()
        if os.environ.get('RENDER') == "t":
            env.render(mode='human')

    steps = 0
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        score = 0
        episode_critic_loss, episode_actor_loss = 0.0, 0.0

        while not done:
            if os.environ.get('RENDER') == "t":
                env.render()

            action = agent.choose_action(state)
            action = np.squeeze(action)
            next_state, reward, done, _, _ = env.step(action)
            steps += 1
            agent.store_transition(state, action, reward, next_state, done)
            if not load_checkpoint:
                update_return = agent.update()  # Capture the return value of agent.update()

                if update_return is not None:  # Check if update_return is not None
                    critic_loss, actor_loss = update_return  # Unpack the values
                    episode_critic_loss += critic_loss
                    episode_actor_loss += actor_loss

            score += reward
            state = next_state

        score_history.append(score)
        mean_100 = np.mean(score_history[-100:])
        avg_score = np.mean(score_history)
        avg_scores.append(avg_score)
        running_mean_100.append(mean_100)
        eps_critic_losses.append(episode_critic_loss)
        eps_actor_losses.append(episode_actor_loss)

        # Record training time for the episode
        end_time = time.time()
        training_time = end_time - start_time
        start_time = end_time
        total_training_time.append(training_time)

        if mean_100 > best_score:
            best_score = mean_100
            agent.save_models()

        if best_score >= success_threshold:
            print(f'100 Episodes Average: {mean_100:.1f}')
            break

        print(f'Episode {episode}: Reward {score:.1f}')


    # Save training log to CSV file
    csv_filename = "training_logs/training_log.csv"
    with open(csv_filename, "w+") as wf:
        writer = csv.writer(wf)
        writer.writerow(['Episode','Reward', 'Average Score', '100-episode Running Mean','Critic Loss', 'Actor Loss', 'Training Time'])
        for idx in range(len(score_history)):
            writer.writerow([idx, score_history[idx], avg_scores[idx], running_mean_100[idx], eps_critic_losses[idx], eps_actor_losses[idx], total_training_time[idx]])

if __name__ == '__main__':
    train_agent()
